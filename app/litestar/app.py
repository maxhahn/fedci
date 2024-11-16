from dataclasses import dataclass, asdict
from enum import Enum

from typing import List, Dict, Set, Optional, Tuple, Union

from litestar import Litestar, MediaType, Response, post, get
from litestar.exceptions import HTTPException
from litestar.params import Body

import datetime
import uuid

from collections import OrderedDict
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import pickle
import base64

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fedci

import numpy as np
import base64
import json

def deserialize_numpy_array(serialized_data):
    data = json.loads(serialized_data)
    arr_base64 = data['data']
    arr_bytes = base64.b64decode(arr_base64)
    arr = np.frombuffer(arr_bytes, dtype=data['dtype']).reshape(data['shape'])
    return arr

def serialize_numpy_array(arr):
    # Convert NumPy array to bytes
    arr_bytes = arr.tobytes()
    # Encode bytes to base64 string
    arr_base64 = base64.b64encode(arr_bytes).decode('utf-8')
    # Create a JSON-serializable dictionary
    data = {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'data': arr_base64
    }
    return json.dumps(data)



#import fedci as fedci


# ,------.          ,--.  ,--.  ,--.  ,--.
# |  .---',--,--, ,-'  '-.`--',-'  '-.`--' ,---.  ,---.
# |  `--, |      \'-.  .-',--.'-.  .-',--.| .-. :(  .-'
# |  `---.|  ||  |  |  |  |  |  |  |  |  |\   --..-'  `)
# `------'`--''--'  `--'  `--'  `--'  `--' `----'`----'


class Algorithm(str, Enum):
    P_VALUE_AGGREGATION = 'IOD'
    FEDERATED_GLM = 'FEDGLM'

class AlgorithmState(str, Enum):
    RUNNING = 'RUNNING'
    FIX_CATEGORICALS = 'FIX_CATEGORICALS'
    FIX_ORDINALS = 'FIX_ORDINALS'
    FINISHED = 'FINISHED'

@dataclass
class Connection:
    id: str
    username: str
    last_request_time: datetime.datetime
    algorithm: Algorithm
    data_labels: List[str]
    categorical_expressions: Dict[str, List[str]]
    ordinal_expressions: Dict[str, List[str]]
    data: Optional[pd.DataFrame]=None

@dataclass
class FederatedGLMData:
    xwx: np.typing.NDArray
    xwz: np.typing.NDArray
    dev: float
    llf: float
    rss: float
    nobs: int

    def __init__(self, fedglm_data_dto):
        self.xwx = deserialize_numpy_array(fedglm_data_dto.xwx)
        self.xwz = deserialize_numpy_array(fedglm_data_dto.xwz)
        self.dev = fedglm_data_dto.dev
        self.llf = fedglm_data_dto.llf
        self.rss = fedglm_data_dto.rss
        self.nobs = fedglm_data_dto.nobs

@dataclass
class FederatedGLMDataDTO:
    xwx: object
    xwz: object
    dev: float
    llf: float
    rss: float
    nobs: int

@dataclass
class NPArray:
    shape: Tuple[int]
    dtype: str
    data: str

@dataclass
class FederatedGLMTesting:
    testing_engine: fedci.TestingEngine
    pending_data: Dict[str, FederatedGLMData]
    start_of_last_iteration: datetime.datetime

@dataclass
class FederatedGLMFixupTesting(FederatedGLMTesting):
    fixup_engine: object

    def __init__(self, testing_engine: fedci.TestingEngine, room, variable_type):
        self.testing_engine = testing_engine

        ce, rce, oe, roe = get_categorical_and_ordinal_expressions_with_reverse(room)

        if variable_type == 'categorical':
            self.fixup_engine = fedci.FixupCategoricalsEngine(testing_engine,
                                                              room.user_provided_labels,
                                                              room.user_provided_categorical_expressions,
                                                              ce,
                                                              rce,
                                                              roe
                                                              )
            pass
        elif variable_type == 'ordinal':
            self.fixup_engine = fedci.FixupOrdinalsEngine(testing_engine,
                                                          room.user_provided_labels,
                                                          room.user_provided_categorical_expressions,
                                                          oe,
                                                          rce,
                                                          roe
                                                          )
        else:
            assert False, f'Unknown variable type: {variable_type}'

        if self.fixup_engine.get_current_test() is None:
            raise Exception('Should not be called, when there are no tests to fix')
        self.pending_data = self.fixup_engine.get_current_test()[1]
        self.start_of_last_iteration = datetime.datetime.now()


@dataclass
class Room:
    name: str
    algorithm: Algorithm
    algorithm_state: AlgorithmState
    owner_name: str
    password: str
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    users: Set[str]
    user_provided_labels: Dict[str, List[str]]
    user_provided_categorical_expressions: Dict[str, Dict[str, List[str]]]
    user_provided_ordinal_expressions: Dict[str, Dict[str, List[str]]]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    user_results: Dict[str, List[List[int]]]
    user_labels: Dict[str, List[str]]
    federated_glm: FederatedGLMTesting


# ,------. ,--------. ,-----.
# |  .-.  \'--.  .--''  .-.  ' ,---.
# |  |  \  :  |  |   |  | |  |(  .-'
# |  '--'  /  |  |   '  '-'  '.-'  `)
# `-------'   `--'    `-----' `----'

@dataclass
class UserDTO:
    id: str
    username: str
    algorithm: str
    data_labels: Optional[List[str]]
    def __init__(self, conn: Connection):
        self.id = conn.id
        self.username = conn.username
        self.algorithm = conn.algorithm.value
        self.data_labels = conn.data_labels

# TODO: add flag for regular operation and for llf correction for ordinal and categorical data
@dataclass
class FederatedGLMStatus:
    y_label: str
    X_labels: List[str]
    is_awaiting_response: bool
    current_beta: object # bytes?
    current_iteration: int
    current_relative_change_in_deviance: float
    start_of_last_iteration: datetime.datetime

    def __init__(self, glm_testing_state: FederatedGLMTesting, requesting_user: str):
        testing_round: fedci.TestingRound = glm_testing_state.testing_engine.testing_rounds[0] # todo: might cause errors
        self.y_label = testing_round.y_label
        self.X_labels = testing_round.X_labels
        self.is_awaiting_response = requesting_user is not None and requesting_user in glm_testing_state.pending_data and glm_testing_state.pending_data.get(requesting_user) is None
        self.current_beta = serialize_numpy_array(testing_round.beta)
        self.current_iteration = testing_round.iterations
        self.current_relative_change_in_deviance = testing_round.get_relative_change_in_deviance()
        self.start_of_last_iteration = glm_testing_state.start_of_last_iteration

@dataclass
class FederatedGLMFixupStatus(FederatedGLMStatus):
    y_label_compare: Optional[str]
    current_beta_compare: Optional[object]

    def __init__(self, glm_testing_state: FederatedGLMFixupTesting, requesting_user: str):
        fixup_round = glm_testing_state.fixup_engine.get_current_test()

        if fixup_round is None:
            raise Exception('Dont call this when engine is done')

        _, _, (X_labels, y_label0, y_label1, beta0, beta1) = fixup_round

        self.y_label = y_label0
        self.current_beta = serialize_numpy_array(beta0)

        self.y_label_compare = y_label1
        self.current_beta_compare = serialize_numpy_array(beta1) if beta1 is not None else None

        self.X_labels = X_labels

        self.is_awaiting_response = requesting_user is not None and requesting_user in glm_testing_state.pending_data and glm_testing_state.pending_data.get(requesting_user) is None
        self.start_of_last_iteration = glm_testing_state.start_of_last_iteration

@dataclass
class RoomDTO:
    name: str
    algorithm: Algorithm
    algorithm_state: AlgorithmState
    owner_name: str
    is_locked: bool
    is_protected: bool
    def __init__(self, room: Room):
        self.name = room.name
        self.algorithm = room.algorithm
        self.algorithm_state = room.algorithm_state
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        self.is_protected = room.password is not None

@dataclass
class RoomDetailsDTO:
    name: str
    algorithm: Algorithm
    algorithm_state: AlgorithmState
    owner_name: str
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    is_protected: bool
    users: List[str]
    user_provided_labels: Dict[str, List[str]]
    user_provided_categorical_expressions: Dict[str, List[str]]
    user_provided_ordinal_expressions: Dict[str, List[str]]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    private_result: List[List[int]]
    private_labels: List[str]
    federated_glm_status: FederatedGLMStatus

    def __init__(self, room: Room, requesting_user: str=None):
        self.name = room.name
        self.algorithm = room.algorithm
        self.algorithm_state = room.algorithm_state
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        self.is_hidden = room.is_hidden
        self.is_processing = room.is_processing
        self.is_finished = room.is_finished
        self.is_protected = room.password is not None
        self.users = sorted(list(room.users))
        self.user_provided_labels = room.user_provided_labels

        # TODO: prevent repeated calculation of this
        categorical_expressions = {}
        for expressions in room.user_provided_categorical_expressions.values():
            for k,v in expressions.items():
                categorical_expressions[k] = sorted(list(set(categorical_expressions.get(k, [])).union(set(v))))
        ordinal_expressions = {}
        for expressions in room.user_provided_ordinal_expressions.values():
            for k,v in expressions.items():
                ordinal_expressions[k] = sorted(list(set(ordinal_expressions.get(k, [])).union(set(v))), key=lambda x: int(x.split('__ord__')[-1]))

        self.user_provided_categorical_expressions = categorical_expressions
        self.user_provided_ordinal_expressions = ordinal_expressions

        self.result = room.result
        self.result_labels = room.result_labels
        self.private_result = room.user_results[requesting_user] if room.user_results is not None and requesting_user in room.user_results else None
        self.private_labels = room.user_labels[requesting_user] if room.user_labels is not None and requesting_user in room.user_labels else None
        if room.federated_glm is None or room.algorithm_state == AlgorithmState.FINISHED:
            self.federated_glm_status = None
        elif room.algorithm_state == AlgorithmState.RUNNING:
            self.federated_glm_status = FederatedGLMStatus(room.federated_glm, requesting_user)
        else:
            self.federated_glm_status = FederatedGLMFixupStatus(room.federated_glm, requesting_user)

# ,------.                                      ,--.
# |  .--. ' ,---.  ,---. ,--.,--. ,---.  ,---.,-'  '-. ,---.
# |  '--'.'| .-. :| .-. ||  ||  || .-. :(  .-''-.  .-'(  .-'
# |  |\  \ \   --.' '-' |'  ''  '\   --..-'  `) |  |  .-'  `)
# `--' '--' `----' `-|  | `----'  `----'`----'  `--'  `----'
#                    `--'

@dataclass
class CheckInRequest:
    username: str
    algorithm: str
    data_labels: List[str]
    categorical_expressions: Dict[str, List[str]]
    ordinal_expressions: Dict[str, List[str]]

@dataclass
class BasicRequest:
    id: str
    username: str

@dataclass
class UpdateUserRequest(BasicRequest):
    algorithm: str
    new_username: str

@dataclass
class RoomCreationRequest(BasicRequest):
    room_name: str
    algorithm: str
    password: str | None

@dataclass
class JoinRoomRequest(BasicRequest):
    password: str | None

@dataclass
class DataSubmissionRequest(BasicRequest):
    data: str

@dataclass
class IODExecutionRequest(BasicRequest):
    alpha: float


@dataclass
class BaseFedGLMRequest(BasicRequest):
    current_beta: object
    current_iteration: int
    data: FederatedGLMDataDTO

@dataclass
class FixupFedGLMRequest(BasicRequest):
    llf: float


# ,------.            ,--.               ,---.   ,--.                         ,--.
# |  .-.  \  ,--,--.,-'  '-. ,--,--.    '   .-',-'  '-.,--.--.,--.,--. ,---.,-'  '-.,--.,--.,--.--. ,---.  ,---.
# |  |  \  :' ,-.  |'-.  .-'' ,-.  |    `.  `-.'-.  .-'|  .--'|  ||  || .--''-.  .-'|  ||  ||  .--'| .-. :(  .-'
# |  '--'  /\ '-'  |  |  |  \ '-'  |    .-'    | |  |  |  |   '  ''  '\ `--.  |  |  '  ''  '|  |   \   --..-'  `)
# `-------'  `--`--'  `--'   `--`--'    `-----'  `--'  `--'    `----'  `---'  `--'   `----' `--'    `----'`----'

rooms: Dict[str, Room] = {}
connections: Dict[str, Connection] = {}
user2room: Dict[str, Room] = {}
user2connection: Dict[str, Connection] = {}
last_cleanse_time = datetime.datetime.now()

# ,--.  ,--.       ,--.
# |  '--'  | ,---. |  | ,---.  ,---. ,--.--. ,---.
# |  .--.  || .-. :|  || .-. || .-. :|  .--'(  .-'
# |  |  |  |\   --.|  || '-' '\   --.|  |   .-'  `)
# `--'  `--' `----'`--'|  |-'  `----'`--'   `----'
#                      `--'

def validate_user_request(id: str, username: str):
    if id not in connections:
        return False

    if connections[id].username != username:
        return False

    connections[id].last_request_time = datetime.datetime.now()
    return True

def cleanse_inactive_users(curr_time):
    for id, conn in connections.items():
        if (curr_time-conn.last_request_time).total_seconds() > 60*60*3:
            username = conn.username
            # if user is in room
            if username in user2room:
                # remove from room
                user2room[username].users.remove(username)
                # if room is empty -> remove room
                if len(user2room[username].users) == 0:
                    del rooms[user2room[username]]
                del user2room[username]
            # remove connection lookup and connection
            del user2connection[username]
            del conn[id]

# ,--.  ,--.               ,--.  ,--.  ,--.          ,-----.,--.                  ,--.
# |  '--'  | ,---.  ,--,--.|  |,-'  '-.|  ,---.     '  .--./|  ,---.  ,---.  ,---.|  |,-.
# |  .--.  || .-. :' ,-.  ||  |'-.  .-'|  .-.  |    |  |    |  .-.  || .-. :| .--'|     /
# |  |  |  |\   --.\ '-'  ||  |  |  |  |  | |  |    '  '--'\|  | |  |\   --.\ `--.|  \  \
# `--'  `--' `----' `--`--'`--'  `--'  `--' `--'     `-----'`--' `--' `----' `---'`--'`--'

@get("/health-check")
async def health_check() -> Response:
    global last_cleanse_time
    curr_time = datetime.datetime.now()
    if (curr_time - last_cleanse_time).total_seconds() > 60*20:
        cleanse_inactive_users(curr_time)
        last_cleanse_time = curr_time

    return Response(
        media_type=MediaType.TEXT,
        content='Hello there!',
        status_code=200
    )

#                                                                                                         ,--.
#  ,-----.,--.,--.                 ,--.       ,-----.,--.                  ,--.          ,--.            /  /,------.
# '  .--./|  |`--' ,---. ,--,--, ,-'  '-.    '  .--./|  ,---.  ,---.  ,---.|  |,-.,-----.|  |,--,--,    /  / |  .--. ' ,---. ,--,--,  ,--,--.,--,--,--. ,---.
# |  |    |  |,--.| .-. :|      \'-.  .-'    |  |    |  .-.  || .-. :| .--'|     /'-----'|  ||      \  /  /  |  '--'.'| .-. :|      \' ,-.  ||        || .-. :
# '  '--'\|  ||  |\   --.|  ||  |  |  |      '  '--'\|  | |  |\   --.\ `--.|  \  \       |  ||  ||  | /  /   |  |\  \ \   --.|  ||  |\ '-'  ||  |  |  |\   --.
#  `-----'`--'`--' `----'`--''--'  `--'       `-----'`--' `--' `----' `---'`--'`--'      `--'`--''--'/  /    `--' '--' `----'`--''--' `--`--'`--`--`--' `----'
#                                                                                                   `--'

@post("/check-in")
async def check_in(data: CheckInRequest) -> Response:

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

    conn = Connection(new_id,
                      new_username,
                      datetime.datetime.now(),
                      algorithm=Algorithm(data.algorithm),
                      data_labels=data.data_labels,
                      categorical_expressions=data.categorical_expressions,
                      ordinal_expressions=data.ordinal_expressions
                      )
    connections[new_id] = conn
    user2connection[new_username] = conn

    return Response(
        media_type=MediaType.JSON,
        content=UserDTO(conn),
        status_code=200
        )

@post("/update-user")
async def update_user(data: UpdateUserRequest) -> Response:
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

#                                                                          ,--.
#  ,----.            ,--.      ,------.                                   /  /,------.                             ,------.           ,--.          ,--.,--.
# '  .-./    ,---. ,-'  '-.    |  .--. ' ,---.  ,---. ,--,--,--. ,---.   /  / |  .--. ' ,---.  ,---. ,--,--,--.    |  .-.  \  ,---. ,-'  '-. ,--,--.`--'|  | ,---.
# |  | .---.| .-. :'-.  .-'    |  '--'.'| .-. || .-. ||        |(  .-'  /  /  |  '--'.'| .-. || .-. ||        |    |  |  \  :| .-. :'-.  .-'' ,-.  |,--.|  |(  .-'
# '  '--'  |\   --.  |  |      |  |\  \ ' '-' '' '-' '|  |  |  |.-'  `)/  /   |  |\  \ ' '-' '' '-' '|  |  |  |    |  '--'  /\   --.  |  |  \ '-'  ||  ||  |.-'  `)
#  `------'  `----'  `--'      `--' '--' `---'  `---' `--`--`--'`----'/  /    `--' '--' `---'  `---' `--`--`--'    `-------'  `----'  `--'   `--`--'`--'`--'`----'
#                                                                    `--'

# list rooms
@post("/rooms")
async def get_rooms(data: BasicRequest) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)

    return Response(
        media_type=MediaType.JSON,
        content=[RoomDTO(room) for room in rooms.values() if not room.is_hidden],
        status_code=200
        )

# get room info
@post("/rooms/{room_name:str}")
async def get_room(data: BasicRequest, room_name: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)

    room = rooms[room_name]

    if data.username not in room.users:
        raise HTTPException(detail='You are not in this room', status_code=403)

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# ,------.                             ,--.          ,--.                                ,--.  ,--.
# |  .--. ' ,---.  ,---. ,--,--,--.    |  |,--,--, ,-'  '-. ,---. ,--.--. ,--,--. ,---.,-'  '-.`--' ,---. ,--,--,
# |  '--'.'| .-. || .-. ||        |    |  ||      \'-.  .-'| .-. :|  .--'' ,-.  || .--''-.  .-',--.| .-. ||      \
# |  |\  \ ' '-' '' '-' '|  |  |  |    |  ||  ||  |  |  |  \   --.|  |   \ '-'  |\ `--.  |  |  |  |' '-' '|  ||  |
# `--' '--' `---'  `---' `--`--`--'    `--'`--''--'  `--'   `----'`--'    `--`--' `---'  `--'  `--' `---' `--''--'

# create room
@post("/rooms/create")
async def create_room(data: RoomCreationRequest) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if len(data.room_name) == 0:
        raise HTTPException(detail='The room must have a name with at least 1 character', status_code=400)

    room_owner = data.username
    room_name = data.room_name

    room_name_offset = 1
    new_room_name = room_name
    occupied_room_names = [r.name for r in rooms.values()]
    while new_room_name in occupied_room_names:
        new_room_name = room_name + f' ({room_name_offset})'
        room_name_offset += 1

    room = Room(name=new_room_name,
                algorithm=Algorithm(data.algorithm),
                algorithm_state=AlgorithmState.RUNNING,
                owner_name=room_owner,
                password=data.password,
                is_locked=False,
                is_hidden=False,
                is_processing=False,
                is_finished=False,
                users={room_owner},
                user_provided_labels={room_owner: connections[data.id].data_labels},
                user_provided_categorical_expressions={room_owner: connections[data.id].categorical_expressions},
                user_provided_ordinal_expressions={room_owner: connections[data.id].ordinal_expressions},
                result=None,
                result_labels=None,
                user_results={},
                user_labels={},
                federated_glm=None
                )

    rooms[new_room_name] = room
    user2room[room_owner] = room

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# join room
@post("/rooms/{room_name:str}/join")
async def join_room(data: JoinRoomRequest, room_name: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if data.username in user2room:
        raise HTTPException(detail='You are already assigned a room', status_code=403)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)

    room = rooms[room_name]

    if room.algorithm == Algorithm.P_VALUE_AGGREGATION and connections[data.id].data is None:
        raise HTTPException(detail='You have to upload data before joining a room', status_code=403)


    if room.is_locked:
        raise HTTPException(detail='The room is locked', status_code=403)
    if room.password != data.password:
        raise HTTPException(detail='Incorrect password', status_code=403)

    room.users.add(data.username)
    room.user_provided_categorical_expressions[data.username] = connections[data.id].categorical_expressions
    room.user_provided_ordinal_expressions[data.username] = connections[data.id].ordinal_expressions
    room.user_provided_labels[data.username] = connections[data.id].data_labels
    user2room[data.username] = room

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# leave room
@post("/rooms/{room_name:str}/leave")
async def leave_room(data: BasicRequest, room_name: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)

    room = rooms[room_name]

    if data.username not in room.users:
        raise HTTPException(detail='You are not in the room', status_code=403)

    room.users.remove(data.username)
    del user2room[data.username]
    # just remove room if it is empty
    if len(room.users) == 0:
        del rooms[room_name]
        return Response(
            media_type=MediaType.TEXT,
            content="You left the room",
            status_code=200
            )

    del room.user_provided_labels[data.username]
    del room.user_provided_categorical_expressions[data.username]
    del room.user_provided_ordinal_expressions[data.username]
    if room.is_finished:
        del room.user_results[data.username]

    if data.username == room.owner_name:
        room.owner_name = room.users[0]

    rooms[room_name] = room # reassign

    return Response(
        media_type=MediaType.TEXT,
        content="You left the room",
        status_code=200
        )

# kick user from room
@post("/rooms/{room_name:str}/kick/{username_to_kick:str}")
async def kick_user_from_room(data: BasicRequest, room_name: str, username_to_kick: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)

    room = rooms[room_name]
    if room.is_finished:
        raise HTTPException(detail='Cannot kick users once IOD was executed', status_code=403)
    if room.is_processing:
        raise HTTPException(detail='Cannot kick while server processes the rooms data', status_code=403)
    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != data.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)
    if room.owner_name == username_to_kick:
        raise HTTPException(detail='Cannot kick the owner of the room', status_code=403)

    if username_to_kick not in room.users:
        raise HTTPException(detail='The person attempted to kick is not inside the room', status_code=403)

    room.users.remove(username_to_kick)
    del room.user_provided_labels[username_to_kick]
    del room.user_provided_categorical_expressions[username_to_kick]
    del room.user_provided_ordinal_expressions[username_to_kick]
    if room.is_finished:
        del room.user_results[username_to_kick]
    del user2room[username_to_kick]
    rooms[room_name] = room

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )


#   ,-.,--. ,--.        ,-.  ,--.                ,--.        ,------.
#  / .'|  | |  |,--,--, '. \ |  |    ,---.  ,---.|  |,-.     |  .--. ' ,---.  ,---. ,--,--,--.
# |  | |  | |  ||      \ |  ||  |   | .-. || .--'|     /     |  '--'.'| .-. || .-. ||        |
# |  | '  '-'  '|  ||  | |  ||  '--.' '-' '\ `--.|  \  \     |  |\  \ ' '-' '' '-' '|  |  |  |
#  \ '. `-----' `--''--'.' / `-----' `---'  `---'`--'`--'    `--' '--' `---'  `---' `--`--`--'
#   `-'                 `-'

def change_room_lock_state(data: BasicRequest, room_name: str, new_lock_state: bool):
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    room = rooms[room_name]

    if room.is_finished:
        raise HTTPException(detail='The room can no longer be changed', status_code=403)

    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != data.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)

    room.is_locked = new_lock_state
    rooms[room_name] = room

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# lock room
@post("/rooms/{room_name:str}/lock")
async def lock_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_lock_state(data, room_name, True)

# lock room
@post("/rooms/{room_name:str}/unlock")
async def unlock_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_lock_state(data, room_name, False)

#                                  ,--.
# ,--.  ,--.,--.   ,--.           /  /,------.                               ,--.    ,------.
# |  '--'  |`--' ,-|  | ,---.    /  / |  .--. ' ,---.,--.  ,--.,---.  ,--,--.|  |    |  .--. ' ,---.  ,---. ,--,--,--.
# |  .--.  |,--.' .-. || .-. :  /  /  |  '--'.'| .-. :\  `'  /| .-. :' ,-.  ||  |    |  '--'.'| .-. || .-. ||        |
# |  |  |  ||  |\ `-' |\   --. /  /   |  |\  \ \   --. \    / \   --.\ '-'  ||  |    |  |\  \ ' '-' '' '-' '|  |  |  |
# `--'  `--'`--' `---'  `----'/  /    `--' '--' `----'  `--'   `----' `--`--'`--'    `--' '--' `---'  `---' `--`--`--'
#                            `--'

def change_room_hidden_state(data: BasicRequest, room_name: str, new_hidden_state: bool):
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)

    room = rooms[room_name]

    if room.is_finished:
        raise HTTPException(detail='The room can no longer be changed', status_code=403)

    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != data.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)

    room.is_hidden = new_hidden_state
    rooms[room_name] = room

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# hide room
@post("/rooms/{room_name:str}/hide")
async def hide_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_hidden_state(data, room_name, True)

# reveal room
@post("/rooms/{room_name:str}/reveal")
async def reveal_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_hidden_state(data, room_name, False)

#  ,-----.,--.,--.                 ,--.      ,------.            ,--.              ,--. ,--.       ,--.                  ,--.
# '  .--./|  |`--' ,---. ,--,--, ,-'  '-.    |  .-.  \  ,--,--.,-'  '-. ,--,--.    |  | |  | ,---. |  | ,---.  ,--,--. ,-|  |
# |  |    |  |,--.| .-. :|      \'-.  .-'    |  |  \  :' ,-.  |'-.  .-'' ,-.  |    |  | |  || .-. ||  || .-. |' ,-.  |' .-. |
# '  '--'\|  ||  |\   --.|  ||  |  |  |      |  '--'  /\ '-'  |  |  |  \ '-'  |    '  '-'  '| '-' '|  |' '-' '\ '-'  |\ `-' |
#  `-----'`--'`--' `----'`--''--'  `--'      `-------'  `--`--'  `--'   `--`--'     `-----' |  |-' `--' `---'  `--`--' `---'
#                                                                                           `--'

# post data to room
@post("/submit-data")
async def receive_data(data: DataSubmissionRequest) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)

    # data to pandas conversion
    connections[data.id].data = pickle.loads(base64.b64decode(data.data.encode()))

    #if data.data_labels is None or data.categorical_expressions is None or data.ordinal_expressions is None:
    #    raise HTTPException(detail='Invalid data provided', status_code=400)

    return Response(
        media_type=MediaType.TEXT,
        content='Data received',
        status_code=200
        )

# ,------.                     ,--. ,-----. ,------.
# |  .--. ',--.,--.,--,--,     |  |'  .-.  '|  .-.  \
# |  '--'.'|  ||  ||      \    |  ||  | |  ||  |  \  :
# |  |\  \ '  ''  '|  ||  |    |  |'  '-'  '|  '--'  /
# `--' '--' `----' `--''--'    `--' `-----' `-------'

def run_riod(data, alpha):
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

def run_iod(data, room_name):
    room = rooms[room_name]

    # gather data of all participants
    participant_data = []
    participant_data_labels = []
    participants = room.users
    for user in participants:
        conn = user2connection[user]
        participant_data.append(conn.data)
        participant_data_labels.append(conn.data_labels)

    return run_riod(zip(participants, participant_data, participant_data_labels), alpha=data.alpha)


def _provide_fed_glm_data(data, room_name) -> Response:
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

@post("/rooms/{room_name:str}/federated-glm-data-fixup")
def provide_fed_glm_data_fixup(data: FixupFedGLMRequest, room_name: str) -> Response:
    return _provide_fed_glm_data(data, room_name)

# TODO: set max regressors
# todo: should lock written data
# todo: verify if object is updated by reference or by value - reassigning to dict may not be required
@post("/rooms/{room_name:str}/federated-glm-data")
def provide_fed_glm_data(data: BaseFedGLMRequest, room_name: str) -> Response:
    return _provide_fed_glm_data(data, room_name)

# TODO: Make room details have more cols for owner - kick col and new avg. response time col

def get_categorical_and_ordinal_expressions_with_reverse(room):
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

def get_base_labels(labels, rev_cat_exp, rev_ord_exp):
    result = []
    for label in labels:
        if label in rev_cat_exp:
            result.append(rev_cat_exp[label])
        elif label in rev_ord_exp:
            result.append(rev_ord_exp[label])
        else:
            result.append(label)
    return result

def run_fed_glm(room_name):
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

@post("/rooms/{room_name:str}/run")
async def run(data: IODExecutionRequest, room_name: str) -> Response:
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
            result, result_labels, user_result, user_labels = run_iod(data, room_name)
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

        run_fed_glm(room_name)

    else:
        raise HTTPException(detail=f'Cannot run room of type {room.algorithm}', status_code=500)

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# ,------.                  ,--.                     ,---.           ,--.
# |  .--. ' ,---. ,--.,--.,-'  '-. ,---. ,--.--.    '   .-'  ,---. ,-'  '-.,--.,--. ,---.
# |  '--'.'| .-. ||  ||  |'-.  .-'| .-. :|  .--'    `.  `-. | .-. :'-.  .-'|  ||  || .-. |
# |  |\  \ ' '-' ''  ''  '  |  |  \   --.|  |       .-'    |\   --.  |  |  '  ''  '| '-' '
# `--' '--' `---'  `----'   `--'   `----'`--'       `-----'  `----'  `--'   `----' |  |-'
#                                                                                  `--'

app = Litestar([
    health_check,
    update_user,
    check_in,
    create_room,
    get_rooms,
    get_room,
    join_room,
    leave_room,
    kick_user_from_room,
    receive_data,
    lock_room,
    unlock_room,
    hide_room,
    reveal_room,
    provide_fed_glm_data,
    provide_fed_glm_data_fixup,
    run
    ])
