from enum import Enum
from dataclasses import dataclass
import datetime
from typing import Union, Dict, List, Set
import pandas as pd
import numpy as np
from typing import Optional
from ls_helpers import deserialize_numpy_array, serialize_numpy_array

import fedci

class Algorithm(str, Enum):
    P_VALUE_AGGREGATION = 'IOD'
    FEDERATED_GLM = 'FEDGLM'

@dataclass
class RIODUserData():
    data_labels: List[str]
    data: pd.DataFrame | None

@dataclass
class FEDGLMUserData():
    data_labels: List[str]
    schema: Dict[str, fedci.VariableType]
    categorical_expressions: Dict[str, List[str]]
    ordinal_expressions: Dict[str, List[str]]

@dataclass
class Connection:
    id: str
    username: str
    last_request_time: datetime.datetime
    algorithm: Algorithm
    provided_data: Union[RIODUserData, FEDGLMUserData]

@dataclass
class FEDGLMUpdateDataDTO:
    xwx: Dict[str, object]
    xwz: Dict[str, object]
    dev: float
    llf: float

@dataclass
class FEDGLMUpdateData:
    xwx: Dict[str, np.typing.NDArray]
    xwz: Dict[str, np.typing.NDArray]
    dev: float
    llf: float

    def __init__(self, data: FEDGLMUpdateDataDTO):
        self.xwx = {c:deserialize_numpy_array(xwx) for c, xwx in data.xwx.items()}
        self.xwz = {c:deserialize_numpy_array(xwz) for c, xwz in data.xwz.items()}
        self.dev = data.dev
        self.llf = data.llf

@dataclass
class FEDGLMState:
    user_provided_schema: Dict[str, Dict[str, fedci.VariableType]]
    user_provided_labels: Dict[str, List[str]]
    user_provided_categorical_expressions: Dict[str, Dict[str, List[str]]]
    user_provided_ordinal_expressions: Dict[str, Dict[str, List[str]]]
    testing_engine: fedci.TestEngine
    pending_data: Dict[str, FEDGLMUpdateData]
    start_of_last_iteration: datetime.datetime

@dataclass
class RIODState:
    user_provided_labels: Dict[str, List[str]]

@dataclass
class Room:
    name: str
    algorithm: Algorithm
    algorithm_state: Union[FEDGLMState, RIODState]
    owner_name: str
    password: str | None
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    users: Set[str]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    user_results: Dict[str, List[List[int]]]
    user_labels: Dict[str, List[str]]

@dataclass
class UserDTO:
    id: str
    username: str
    algorithm: str
    data_labels: List[str]

    def __init__(self, conn: Connection):
        self.id = conn.id
        self.username = conn.username
        self.algorithm = conn.algorithm.value
        self.data_labels = conn.provided_data.data_labels

@dataclass
class RoomDTO:
    name: str
    algorithm: Algorithm
    owner_name: str
    is_locked: bool
    is_protected: bool
    def __init__(self, room: Room):
        self.name = room.name
        self.algorithm = room.algorithm
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        self.is_protected = room.password is not None

@dataclass
class FEDGLMInformation:
    is_awaiting_response: bool
    categorical_expressions: Dict[str, Dict[str, List[str]]]
    ordinal_expressions: Dict[str, Dict[str, List[str]]]
    current_beta: Dict[str, object]
    current_iteration: int
    y_label: str
    X_labels: List[str]

    def __init__(self, testing_engine: fedci.TestEngine, is_awaiting_response: bool):
        self.is_awaiting_response = is_awaiting_response
        self.categorical_expressions = testing_engine.categorical_expressions
        self.ordinal_expressions = testing_engine.ordinal_expressions if fedci.EXPAND_ORDINALS else None
        self.y_label, self.X_labels, self.current_beta = testing_engine.get_current_test_parameters()
        self.current_beta = {k:serialize_numpy_array(v) for k,v in self.current_beta.items()}
        self.current_iteration = testing_engine.get_current_test_iteration()


@dataclass
class RoomDetailsDTO:
    name: str
    algorithm: Algorithm
    algorithm_info: FEDGLMInformation | None
    owner_name: str
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    is_protected: bool
    users: List[str]
    user_provided_labels: Dict[str, List[str]]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    private_result: List[List[int]]
    private_labels: List[str]

    def __init__(self, room: Room, requesting_user: Union[str,None]=None):
        self.name = room.name
        self.algorithm = room.algorithm
        self.algorithm_info = FEDGLMInformation(
            room.algorithm_state.testing_engine,
            requesting_user is not None and requesting_user in room.algorithm_state.pending_data and room.algorithm_state.pending_data.get(requesting_user) is None
        ) if (
            self.algorithm == Algorithm.FEDERATED_GLM and room.algorithm_state.testing_engine is not None and not room.algorithm_state.testing_engine.is_finished()
        ) else None
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        self.is_hidden = room.is_hidden
        self.is_processing = room.is_processing
        self.is_finished = room.is_finished
        self.is_protected = room.password is not None
        self.users = sorted(list(room.users))
        self.user_provided_labels = room.algorithm_state.user_provided_labels
        self.result = room.result
        self.result_labels = room.result_labels
        self.private_result = room.user_results[requesting_user] if room.user_results is not None and requesting_user in room.user_results else None
        self.private_labels = room.user_labels[requesting_user] if room.user_labels is not None and requesting_user in room.user_labels else None

@dataclass
class CheckInRequest:
    username: str
    algorithm: str
    # riod
    data_labels: List[str] | None
    # fedglm
    schema: object | None
    categorical_expressions: Dict[str, List[str]] | None
    ordinal_expressions: Dict[str, List[str]] | None

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
class RIODDataSubmissionRequest(BasicRequest):
    data: str

@dataclass
class IODExecutionRequest(BasicRequest):
    alpha: float

@dataclass
class UpdateFEDGLMRequest(BasicRequest):
    current_beta: object
    current_iteration: int
    data: FEDGLMUpdateDataDTO
