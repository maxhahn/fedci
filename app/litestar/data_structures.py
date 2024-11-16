from enum import Enum
from dataclasses import dataclass
import datetime
from typing import Optional, Union, Dict, List, Set
import pandas as pd
import numpy as np

from .helpers import deserialize_numpy_array
from .. import fedci

class Algorithm(str, Enum):
    P_VALUE_AGGREGATION = 'IOD'
    FEDERATED_GLM = 'FEDGLM'

@dataclass
class RIODUserData()
    data_labels: List[str]
    data: Optional[pd.DataFrame]

@dataclass
class FEDGLMUserData():
    schema: Dict[str, Dict[str, fedci.VariableType]]
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
    xwx: object
    xwz: object
    dev: float
    llf: float

@dataclass
class FEDGLMUpdateData:
    xwx: np.typing.NDArray
    xwz: np.typing.NDArray
    dev: float
    llf: float

    def __init__(self, data: FEDGLMUpdateDataDTO):
        self.xwx = deserialize_numpy_array(data.xwx)
        self.xwz = deserialize_numpy_array(data.xwz)
        self.dev = data.dev
        self.llf = data.llf

@dataclass
class FEDGLMState:
    schema: Dict[str, fedci.VariableType]
    categorical_expressions: Dict[str, Dict[str, List[str]]]
    ordinal_expressions: Dict[str, Dict[str, List[str]]]
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
    password: Optional[str]
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
class RoomDetailsDTO:
    name: str
    algorithm: Algorithm
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
    categorical_expressions: Optional[Dict[str, List[str]]]
    ordinal_expressions: Optional[Dict[str, List[str]]]

    def __init__(self, room: Room, requesting_user: Union[str,None]=None):
        self.name = room.name
        self.algorithm = room.algorithm
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
        for expressions in room.algorithm_state.user_provided_categorical_expressions.values():
            for k,v in expressions.items():
                categorical_expressions[k] = sorted(list(set(categorical_expressions.get(k, [])).union(set(v))))
        self.categorical_expressions = categorical_expressions

        if fedci.EXPAND_ORDINALS:
            ordinal_expressions = {}
            for expressions in room.algorithm_state.user_provided_ordinal_expressions.values():
                for k,v in expressions.items():
                    ordinal_expressions[k] = sorted(list(set(ordinal_expressions.get(k, [])).union(set(v))), key=lambda x: int(x.split('__ord__')[-1]))
            self.ordinal_expressions = ordinal_expressions

        self.result = room.result
        self.result_labels = room.result_labels
        self.private_result = room.user_results[requesting_user] if room.user_results is not None and requesting_user in room.user_results else None
        self.private_labels = room.user_labels[requesting_user] if room.user_labels is not None and requesting_user in room.user_labels else None

@dataclass
class CheckInRequest:
    username: str
    algorithm: str
    # riod
    data_labels: Optional[List[str]]
    # fedglm
    schema: Optional[object]
    categorical_expressions: Optional[Dict[str, List[str]]]
    ordinal_expressions: Optional[Dict[str, List[str]]]

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
