from litestar import Controller, Response, MediaType, post
from litestar.exceptions import HTTPException
from typing import Optional
from ls_data_structures import IODExecutionRequest, Algorithm, RoomDetailsDTO, FEDGLMState, UpdateFEDGLMRequest, FEDGLMUpdateData
from ls_env import connections, rooms, user2connection
from ls_helpers import validate_user_request

from dataclasses import asdict

import fedci

from collections import OrderedDict
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd

import datetime
from typing import Set

class AlgorithmController(Controller):
    path = '/run'

    # TODO: Missing ord0 rows need to be added with p value 0! (see algorithm 1 of tillman and sprites 2011)
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

    def setup_fedglm(self, room_name, alpha, max_cond_size):
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
            ordinal_expressions=ordinal_expressions,
            max_regressors=max_cond_size
        )

        room.algorithm_state.testing_engine = testing_engine
        room.algorithm_state.alpha = alpha
        room.algorithm_state.start_of_last_iteration=datetime.datetime.now()

        required_labels = testing_engine.get_currently_required_labels()

        pending_data = {client:None for client, labels in room.algorithm_state.user_provided_labels.items() if set(required_labels).issubset(labels)}
        assert len(pending_data) > 0, f'There are no clients who can supply the labels: {required_labels}'
        # TODO: consider making a pending_data queue, so that fast clients can process their data early. But slowness still remains... weakest link
        room.algorithm_state.pending_data = pending_data

        rooms[room_name] = room
        return

    @post('/{room_name:str}/fedglm-data')
    async def receive_fedglm_data(self, data: UpdateFEDGLMRequest, room_name: str) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
        if room_name not in rooms:
            raise HTTPException(detail='The room does not exist', status_code=404)
        room = rooms[room_name]

        if data.username not in room.algorithm_state.pending_data.keys():
            return Response(
                content='The provided data was not required',
                status_code=200
                )

        if type(data) == UpdateFEDGLMRequest and data.current_iteration != room.algorithm_state.testing_engine.get_current_test_iteration():
            return Response(
                content='The provided data was not usable in this iteration',
                status_code=200
                )

        room.algorithm_state.pending_data[data.username] = FEDGLMUpdateData(data.data)

        # If there are still missing entries
        if any([v is None for v in room.algorithm_state.pending_data.values()]):
            rooms[room_name] = room
            return Response(
                    content='The provided data was accepted',
                    status_code=200
                    )

        result = {}
        for client, _data in room.algorithm_state.pending_data.items():
            beta_update = {}
            for c in _data.xwx.keys():
                beta_update[c] = fedci.BetaUpdateData(_data.xwx[c], _data.xwz[c])
            r = fedci.ClientResponseData(_data.llf, _data.dev, beta_update)
            result[client] = r

        room.algorithm_state.testing_engine.update_current_test(result)

        if room.algorithm_state.testing_engine.is_finished():
            likelihood_ratio_tests = fedci.get_symmetric_likelihood_tests(room.algorithm_state.testing_engine.tests)

            all_labels = list(set([li for l in room.algorithm_state.user_provided_labels.values() for li in l]))

            columns = ('ord', 'X', 'Y', 'S', 'pvalue')
            rows = []
            for test in likelihood_ratio_tests:
                s_labels_string = ','.join(sorted([str(all_labels.index(l)+1) for l in test.conditioning_set]))
                rows.append((len(test.conditioning_set), all_labels.index(test.v0)+1, all_labels.index(test.v1)+1, s_labels_string, test.p_val))

            df = pd.DataFrame(data=rows, columns=columns)

            try:
                result, result_labels, _, _ = self.run_riod([(None, df, all_labels)], alpha=room.algorithm_state.alpha)
            except:
                raise HTTPException(detail='Failed to execute IOD', status_code=500)

            room.result = result
            room.result_labels = result_labels

            room.is_processing = False
            room.is_finished = True
        else:
            required_labels = room.algorithm_state.testing_engine.get_currently_required_labels()
            pending_data = {client:None for client, labels in room.algorithm_state.user_provided_labels.items() if set(required_labels).issubset(labels)}
            assert len(pending_data) > 0, f'There are no clients who can supply the labels: {required_labels}'
            room.algorithm_state.pending_data = pending_data


        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200
            )

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
            self.setup_fedglm(room_name, data.alpha, data.max_conditioning_set)
        else:
            raise Exception(f'Encountered unknown algorithm {room.algorithm}')

        return Response(
            media_type=MediaType.JSON,
            content=RoomDetailsDTO(room, data.username),
            status_code=200
            )
