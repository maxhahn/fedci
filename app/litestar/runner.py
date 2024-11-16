from litestar import Controller, Response, MediaType, post
from litestar.exceptions import HTTPException

from app.fedci.env import EXPAND_ORDINALS

from .data_structures import IODExecutionRequest, Algorithm, RoomDetailsDTO, FEDGLMState
from .env import connections, rooms, user2connection
from .helpers import validate_user_request

from .. import fedci

from collections import OrderedDict
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

import datetime
from typing import Set

class AlgorithmController(Controller):
    path = '/run'

    def run_riod(self, data, alpha):
        users = []

        ro.r['source']('./scripts/aggregation.r')
        aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
        # Reading and processing data
        #df = pl.read_csv("./random-data-1.csv")
        with (ro.default_converter + pandas2ri.converter).context():
            lvs = []
            for user, df, labels in data:
                #converting it into r object for passing into r function
                d = [('citestResults', ro.conversion.get_conversion().py2rpy(df)), ('labels', ro.StrVector(labels))]
                od = OrderedDict(d)
                lv = ro.ListVector(od)
                lvs.append(lv)
                users.append(user)

            result = aggregate_ci_results_f(lvs, alpha)

            g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
            g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
            gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_List'].items()]
            gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
            return g_pag_list, g_pag_labels,  {u:r for u,r in zip(users, gi_pag_list)}, {u:l for u,l in zip(users, gi_pag_labels)}

    def run_iod(self, data, room_name):
        room = rooms[room_name]

        # gather data of all participants
        participant_data = []
        participant_data_labels = []
        participants = room.users
        for user in participants:
            conn = user2connection[user]
            participant_data.append(conn.provided_data.data)
            participant_data_labels.append(conn.provided_data.data_labels)

        return self.run_riod(zip(participants, participant_data, participant_data_labels), alpha=data.alpha)

    def run_fedglm(self, room_name):
        room = rooms[room_name]

        schema = {}
        for conn in connections.values():
            conn_schema = conn.provided_data.schema
            for column, dtype in conn_schema.items():
                if column not in schema:
                    schema[column] = dtype
                    continue
                assert schema[column] == dtype, f'Schema mismatch between clients detected for variable {column}!'

        category_expressions = {}
        ordinal_expressions = {}
        for conn in connections.values():
            for feature, expressions in conn.provided_data.categorical_expressions.items():
                category_expressions[feature] = sorted(list(set(category_expressions.get(feature, [])).union(set(expressions))))
            for feature, levels in conn.provided_data.ordinal_expressions.items():
                ordinal_expressions[feature] = sorted(list(set(ordinal_expressions.get(feature, [])).union(set(levels))), key=lambda x: int(x.split('__ord__')[-1]))

        testing_engine = fedci.TestEngine(
            schema=schema,
            category_expressions=category_expressions,
            ordinal_expressions=ordinal_expressions
        )
        required_labels = testing_engine.get_currently_required_labels()

        pending_data = {client:None for client, schema in room.algorithm_data.user_provided_schema.items() if set(required_labels).issubset(schema.keys())}
        assert len(pending_data) > 0, f'There are no clients who can supply the labels: {required_labels}'

        room.federated_glm = FEDGLMState(
            schema=schema,
            categorical_expressions=category_expressions,
            ordinal_expressions=ordinal_expressions,
            testing_engine=testing_engine,
            pending_data=pending_data,
            start_of_last_iteration=datetime.datetime.now()
        )

        # TODO: probide this information in room details
        room.algorithm_state.categorical_expressions = category_expressions
        if EXPAND_ORDINALS:
            room.algorithm_state.ordinal_expressions = ordinal_expressions

        rooms[room_name] = room

        # TODO: This should be all for server setup
        # Make sure required vars are exposed so pending data users know what to calculate
        # Next up: Collect incoming data and execute algorithm
        # NOTE: Many renames might have cause some issues in other files with class and variable names/imports/etc.

        pass

    def _provide_fed_glm_data(self, data, room_name) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
        if room_name not in rooms:
            raise HTTPException(detail='The room does not exist', status_code=404)
        room = rooms[room_name]

        if data.username not in room.federated_glm.pending_data.keys():
            return Response(
                content='The provided data was not required',
                status_code=200
                )

        # dont accept data from non-current iteration
        if type(data) == BaseFedGLMRequest and data.current_iteration != room.federated_glm.testing_engine.get_current_test().iterations:
            return Response(
                content='The provided data was not usable in this iteration',
                status_code=200
                )

        # handle requests depending on algorithm state
        if type(data) == BaseFedGLMRequest and room.algorithm_state == AlgorithmState.RUNNING:
            fedglm_data = FederatedGLMData(data.data)
        elif type(data) == FixupFedGLMRequest and room.algorithm_state != AlgorithmState.RUNNING:
            fedglm_data = data.llf
        else:
            return Response(
                    content='The provided data does not fit the currently requested data',
                    status_code=400
                    )

        room.federated_glm.pending_data[data.username] = fedglm_data

        if any([v is None for v in room.federated_glm.pending_data.values()]):
            rooms[room_name] = room

            return Response(
                    content='The provided data was accepted',
                    status_code=200
                    )

        if type(room.federated_glm) == FederatedGLMTesting:
            current_engine = room.federated_glm.testing_engine
            fedglm_results = {k:tuple(asdict(d).values()) for k,d in room.federated_glm.pending_data.items()}
        elif type(room.federated_glm) == FederatedGLMFixupTesting:
            current_engine = room.federated_glm.fixup_engine
            fedglm_results = room.federated_glm.pending_data # already correct format
        else:
            raise Exception(f'Unknown Testing Engine type: {type(room.federated_glm)}')

        current_engine.aggregate_results(fedglm_results)

        categorical_expressions, reversed_categorical_expressions, ordinal_expressions, reversed_ordinal_expressions = get_categorical_and_ordinal_expressions_with_reverse(room)

        # HANDLE FINISHED ENGINE
        if current_engine.is_finished and room.algorithm_state == AlgorithmState.RUNNING:
            room.federated_glm = FederatedGLMFixupTesting(room.federated_glm.testing_engine, room, 'categorical')
            room.algorithm_state = AlgorithmState.FIX_CATEGORICALS
            current_engine = room.federated_glm.fixup_engine
        if current_engine.is_finished and room.algorithm_state == AlgorithmState.FIX_CATEGORICALS:
            room.federated_glm = FederatedGLMFixupTesting(room.federated_glm.testing_engine, room, 'ordinal')
            room.algorithm_state = AlgorithmState.FIX_ORDINALS
            current_engine = room.federated_glm.fixup_engine
        if current_engine.is_finished and room.algorithm_state == AlgorithmState.FIX_ORDINALS:
            room.algorithm_state = AlgorithmState.FINISHED

        # AS LONG AS STATE NOT EQ TO FINISHED, THE ENGINE WILL PROVIDE A TEST
        if room.algorithm_state == AlgorithmState.RUNNING:
            curr_testing_round = current_engine.get_current_test()

            #print(f'Running {curr_testing_round}')

            # ALL DATA HAS BEEN PROVIDED
            required_labels = curr_testing_round.get_required_labels()
            required_labels = get_base_labels(required_labels, reversed_categorical_expressions, reversed_ordinal_expressions)

            pending_data = {client:None for client, labels in room.user_provided_labels.items() if all([required_label in labels for required_label in required_labels])}

            assert len(pending_data) > 0, f'There are no clients who can supply the labels: {required_labels}'

            room.federated_glm.pending_data = pending_data
            room.federated_glm.start_of_last_iteration = datetime.datetime.now()

        elif room.algorithm_state == AlgorithmState.FIX_CATEGORICALS or room.algorithm_state == AlgorithmState.FIX_ORDINALS:
            _, pending_data, _ = current_engine.get_current_test()
            #print(f'Running fixup {pending_data}')
            room.federated_glm.pending_data = pending_data
            room.federated_glm.start_of_last_iteration = datetime.datetime.now()

        elif room.algorithm_state == AlgorithmState.FINISHED:
            # COULD RUN TESTS ACCORDING TO FCI REQUIREMENTS - requires a lot of work

            # TODO: DO LIKELIHOOD RATIO TESTS
            # TODO: RUN FCI

            # turn linear models into likelihood ratio tests
            likelihood_ratio_tests = fedci.get_test_results(room.federated_glm.testing_engine.finished_rounds,
                                                            categorical_expressions,
                                                            reversed_categorical_expressions,
                                                            ordinal_expressions,
                                                            reversed_ordinal_expressions
                                                            )

            # TODO: BUILD PANDAS DF data with columns for IOD -> X,Y,S,p_value (check again)
            # THEN CALL rIOD

            all_labels = list(set([li for l in room.user_provided_labels.values() for li in l]))

            columns = ('ord', 'X', 'Y', 'S', 'pvalue')
            rows = []
            for test in likelihood_ratio_tests:
                s_labels_string = ','.join(sorted([str(all_labels.index(l)+1) for l in test.s_labels]))
                rows.append((len(test.s_labels), all_labels.index(test.x_label)+1, all_labels.index(test.y_label)+1, s_labels_string, test.p_val))

            df = pd.DataFrame(data=rows, columns=columns)

            # TODO: add alpha configuration per request

            try:
                result, result_labels, _, _ = run_riod([(None, df, all_labels)], alpha=0.05)
            except:
                raise HTTPException(detail='Failed to execute FCI', status_code=500)

            room.result = result
            room.result_labels = result_labels

            room.is_processing = False
            room.is_finished = True
        else:
            raise Exception(f'Unknown State - {room.algorithm_state}')

        rooms[room_name] = room

        return Response(
                content='The provided data was accepted',
                status_code=200
                )

    # TODO: set max regressors
    # todo: should lock written data
    # todo: verify if object is updated by reference or by value - reassigning to dict may not be required
    @post("/rooms/{room_name:str}/federated-glm-data")
    def provide_fed_glm_data(self, data: BaseFedGLMRequest, room_name: str) -> Response:
        return self._provide_fed_glm_data(data, room_name)

    # TODO: Make room details have more cols for owner - kick col and new avg. response time col

    def get_categorical_and_ordinal_expressions_with_reverse(self, room):
        categorical_expressions = {}
        for expressions in room.user_provided_categorical_expressions.values():
            for k,v in expressions.items():
                categorical_expressions[k] = sorted(list(set(categorical_expressions.get(k, [])).union(set(v))))
        ordinal_expressions = {}
        for expressions in room.user_provided_ordinal_expressions.values():
            for k,v in expressions.items():
                ordinal_expressions[k] = sorted(list(set(ordinal_expressions.get(k, [])).union(set(v))))

        reversed_category_expressions = {vi:k for k,v in categorical_expressions.items() for vi in v}
        reversed_ordinal_expressions = {vi:k for k,v in ordinal_expressions.items() for vi in v}

        return categorical_expressions, reversed_category_expressions, ordinal_expressions, reversed_ordinal_expressions

    def get_base_labels(self, labels, rev_cat_exp, rev_ord_exp):
        result = []
        for label in labels:
            if label in rev_cat_exp:
                result.append(rev_cat_exp[label])
            elif label in rev_ord_exp:
                result.append(rev_ord_exp[label])
            else:
                result.append(label)
        return result

    def run_fed_glm(self, room_name):
        # data contains alpha for FCI
        room = rooms[room_name]

        available_labels = set([vi for v in room.user_provided_labels.values() for vi in v])
        categorical_expressions, reversed_categorical_expressions, ordinal_expressions, reversed_ordinal_expressions = get_categorical_and_ordinal_expressions_with_reverse(room)

        testing_engine = fedci.TestingEngine(available_labels, categorical_expressions, ordinal_expressions, max_regressors=None)
        required_labels = testing_engine.get_current_test().get_required_labels()
        required_labels = get_base_labels(required_labels, reversed_categorical_expressions, reversed_ordinal_expressions)

        pending_data = {client:None for client, labels in room.user_provided_labels.items() if all([required_label in labels for required_label in required_labels])}

        assert len(pending_data) > 0, f'There are no clients who can supply the labels: {required_labels}'

        room.federated_glm = FederatedGLMTesting(testing_engine=testing_engine,
                                                pending_data=pending_data,
                                                start_of_last_iteration=datetime.datetime.now()
                                                )

        rooms[room_name] = room

    @post("/{room_name:str}")
    async def run(self, data: IODExecutionRequest, room_name: str) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
        if room_name not in rooms:
            raise HTTPException(detail='The room does not exist', status_code=404)
        room = rooms[room_name]

        # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
        if room.owner_name != data.username:
            raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)

        room.is_processing = True
        room.is_locked = True
        room.is_hidden = True
        rooms[room_name] = room

        # ToDo change behavior based on room type:
        #   one run for pvalue aggregation only
        #   multiple runs for fedglm -> therefore, return in this function quickly. Set 'is_processing' and update FedGLMStatus etc
        if room.algorithm == Algorithm.P_VALUE_AGGREGATION:
            try:
                result, result_labels, user_result, user_labels = self.run_iod(data, room_name)
            except:
                room = rooms[room_name]
                room.is_processing = False
                room.is_locked = True
                room.is_hidden = True
                rooms[room_name] = room
                raise HTTPException(detail='Failed to execute IOD', status_code=500)

            room = rooms[room_name]
            for user in user_result:
                if user not in room.users:
                    del user_result[user]

            room.result = result
            room.result_labels = result_labels
            room.user_results = user_result
            room.user_labels = user_labels
            room.is_processing = False
            room.is_finished = True
            rooms[room_name] = room
        elif room.algorithm == Algorithm.FEDERATED_GLM:
            room.is_processing = True
            room.is_locked = True
            room.is_hidden = True
            rooms[room_name] = room

            self.run_fed_glm(room_name)

        else:
            raise Exception(f'Encountered unknown algorithm {room.algorithm}')

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200
            )
