import enum
from dataclasses import dataclass
from typing import Dict

class VariableType(enum.Enum):
    CONTINUOS = 0
    BINARY = 1
    CATEGORICAL = 2
    ORDINAL = 3
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented
    def __hash__(self):
        return hash(self.value)

@dataclass
class BetaUpdateData:
    xwx: object
    xwz: object

@dataclass
class ClientResponseData:
    llf: float
    deviance: float
    beta_update_data: Dict[str, BetaUpdateData]
