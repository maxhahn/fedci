import enum
from dataclasses import dataclass

import polars as pl


class VariableType(int, enum.Enum):
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


polars_dtype_map = {
    pl.Float64: VariableType.CONTINUOS,
    pl.Boolean: VariableType.BINARY,
    pl.String: VariableType.CATEGORICAL,
    pl.Int32: VariableType.ORDINAL,
    pl.Int64: VariableType.ORDINAL,
}

categorical_separator = "__cat__"
ordinal_separator = "__ord__"
constant_colname = "__const"

import numpy as np


@dataclass
class BetaUpdateData:
    llf: float
    xwx: np.ndarray
    xwz: np.ndarray
