from litestar import Controller, post, Response, MediaType
from litestar.exceptions import HTTPException
from .data_structures import Algorithm, Connection, Room, CheckInRequest, UpdateUserRequest, FEDGLMAlgorithmData, RIODAlgorithmData, UserDTO, RIODDataSubmissionRequest
from .env import connections, user2connection
from .helpers import validate_user_request

import uuid
import datetime
import pickle
import base64


class UserController(Controller):
    path = '/user'

    @post("/check-in")
    async def check_in(self, data: CheckInRequest) -> Response:
        if data.username is None or len(data.username.replace(r'\s', '')) == 0:
            raise HTTPException(detail='Username is not accepted', status_code=400)

        # guarantee unused new id
        new_id = uuid.uuid4()
        while new_id in connections:
            new_id = uuid.uuid4()
        new_id = str(new_id)

        username = data.username

        username_offset = 1
        occupied_usernames = [c.username for c in connections.values()]
        new_username = username
        while new_username in occupied_usernames:
            new_username = username + f' ({username_offset})'
            username_offset += 1

        selected_algorithm = Algorithm(data.algorithm)
        if selected_algorithm == Algorithm.FEDERATED_GLM:
            provided_data = FEDGLMAlgorithmData(
                schema=data.schema,
                categorical_expressions=data.categorical_expressions,
                ordinal_expressions=data.ordinal_expressions
            )
        elif selected_algorithm == Algorithm.P_VALUE_AGGREGATION:
            provided_data = RIODAlgorithmData(
                data_labels = data.data_labels,
                data=None
            )
        else:
            raise Exception(f'Unknown algorithm {selected_algorithm}')

        conn = Connection(new_id,
                        new_username,
                        datetime.datetime.now(),
                        algorithm=selected_algorithm,
                        provided_data=provided_data
                        )
        connections[new_id] = conn
        user2connection[new_username] = conn

        return Response(
            media_type=MediaType.JSON,
            content=UserDTO(conn),
            status_code=200
            )

    @post("/update-user")
    async def update_user(self, data: UpdateUserRequest) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)

        #if data.username is None or len(data.username.replace(r'\s', '')) == 0:
        #    raise HTTPException(detail='Username is not accepted', status_code=400)

        if data.new_username is not None and len(data.new_username.replace(r'\s', '')) > 0:
            username_offset = 1
            occupied_usernames = [c.username for c in connections.values()]
            new_username = data.new_username
            while new_username in occupied_usernames:
                new_username = data.new_username + f' ({username_offset})'
                username_offset += 1

            connections[data.id].username = new_username

        connections[data.id].algorithm = Algorithm(data.algorithm)
        conn = connections[data.id]

        return Response(
            media_type=MediaType.JSON,
            content=UserDTO(conn),
            status_code=200
            )

    @post("/submit-data")
    async def receive_data(self, data: RIODDataSubmissionRequest) -> Response:
        if not validate_user_request(data.id, data.username):
            raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)

        if connections[data.id].algorithm != Algorithm.P_VALUE_AGGREGATION:
            raise HTTPException(detail=f'Incompatible algorithm {connections[data.id].algorithm} for data submission', status_code=403)

        # data to pandas conversion
        connections[data.id].provided_data.data = pickle.loads(base64.b64decode(data.data.encode()))

        #if data.data_labels is None or data.categorical_expressions is None or data.ordinal_expressions is None:
        #    raise HTTPException(detail='Invalid data provided', status_code=400)

        return Response(
            media_type=MediaType.TEXT,
            content='Data received',
            status_code=200
            )
