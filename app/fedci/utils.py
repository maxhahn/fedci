import enum
from dataclasses import dataclass
from typing import Dict

class VariableType(enum.Enum):
    CONTINUOS = 0
    BINARY = 1
    CATEGORICAL = 2
    ORDINAL = 3
    
@dataclass
class BetaUpdateData:
    xwx: object
    xwz: object
    
@dataclass
class ClientResponseData:
    llf: float
    deviance: float
    beta_update_data: Dict[str, BetaUpdateData]
    

    