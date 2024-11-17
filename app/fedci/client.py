

from .utils import VariableType, ClientResponseData, BetaUpdateData
import polars as pl
import numpy as np

from typing import Dict, List

from .env import DEBUG, EXPAND_ORDINALS, LR

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.genmod.families import family

class ComputationHelper():
    @staticmethod
    def get_regression_model(y, X, beta, glm_family):
        model = sm.GLM(y, X, family=glm_family)
        result = GLMResults(model, beta, normalized_cov_params=None, scale=None)
        return result

    @staticmethod
    def run_model(y, X, model):
        llf = model.llf
        deviance = model.deviance

        # calculate fisher information and score vector
        eta = model.predict(which='linear')

        # g' is inverse of link function
        inverse_link = model.family.link.inverse
        mu = inverse_link(eta)

        # delta g' is derivative of inverse link function
        derivative_inverse_link = model.family.link.inverse_deriv
        dmu_deta = derivative_inverse_link(eta)

        z = eta + LR*(y - mu)/dmu_deta
        W = np.diag((dmu_deta**2)/np.nanmax([np.var(mu), 1e-4]))

        xw = X.T @ W
        xwx = xw @ X
        xwz = xw @ z

        return {'llf': llf, 'deviance': deviance, 'xwx': xwx, 'xwz': xwz}

    @classmethod
    def run_regression(cls, y, X, beta, glm_family):
        model = cls.get_regression_model(y, X, beta, glm_family)
        return cls.run_model(y, X, model)

class ComputationUnit():
    @staticmethod
    def compute(data, y_label, X_labels, beta):
        raise NotImplementedError()

class ContinousComputationUnit(ComputationUnit):
    @staticmethod
    def compute(data, y_label, X_labels, beta):
        assert len(beta) == 1, 'Continuos regression called with more than one beta'
        beta = list(beta.values())[0]

        X = data.to_pandas()[sorted(X_labels)]
        X['__const'] = 1
        X = X.to_numpy().astype(float)

        y = data.to_pandas()[y_label]
        y = y.to_numpy().astype(float)

        return ComputationHelper.run_regression(
            y=y,
            X=X,
            beta=beta,
            glm_family=family.Gaussian()
        )

class BinaryComputationUnit(ComputationUnit):
    @staticmethod
    def compute(data, y_label, X_labels, beta):
        assert len(beta) == 1, 'Binary regression called with more than one beta'
        beta = list(beta.values())[0]

        X = data.to_pandas()[sorted(X_labels)]
        X['__const'] = 1
        X = X.to_numpy().astype(float)

        y = data.to_pandas()[y_label]
        y = y.to_numpy().astype(float)

        X = data.to_pandas()[sorted(X_labels)]
        X['__const'] = 1
        X = X.to_numpy().astype(float)

        y = data.to_pandas()[y_label]
        y = y.to_numpy().astype(float)

        return ComputationHelper.run_regression(
            y=y,
            X=X,
            beta=beta,
            glm_family=family.Binomial()
        )

class CategoricalComputationUnit(ComputationUnit):
    @staticmethod
    def compute(data, y_label, X_labels, betas):
        X = data.to_pandas()[sorted(X_labels)]
        X['__const'] = 1
        X = X.to_numpy().astype(float)

        models = {}
        for category in betas.keys():
            y = data.to_pandas()[category]
            y = y.to_numpy().astype(float)

            models[category] = ComputationHelper.get_regression_model(
                y=y,
                X=X,
                beta=betas[category],
                glm_family=family.Binomial()
            )

        etas = {c:np.clip(m.predict(which='linear'), -350, 350) for c,m in models.items()}
        denom = 1 + sum(np.exp(eta) for eta in etas.values())
        mus = {c:np.clip(np.exp(eta)/denom,1e-15,1-1e-15) for c,eta in etas.items()}
        dmu_deta = {c:mu*(1-mu) for c,mu in mus.items()}

        results = {}
        llf = 0
        llf_saturated = 0
        reference_category_indices = np.ones(len(data))
        for category in dmu_deta.keys():
            y = data.to_pandas()[category]
            y = y.to_numpy().astype(float)

            reference_category_indices = reference_category_indices * (y==0)

            # beta update
            z = etas[category] + (y - mus[category])/dmu_deta[category]
            W = np.diag((dmu_deta[category]**2)/np.nanmax([np.var(mus[category]), 1e-4]))

            xw = X.T @ W
            xwx = xw @ X
            xwz = xw @ z

            results[category] = {'xwx': xwx, 'xwz': xwz}

            # LLF
            llf += np.sum(np.log(np.take(mus[category], np.nonzero(y)[0])))
            ## LLF SATURATED (for deviance)
            #llf_saturated += np.sum(y * np.log(np.clip(y, 1e-10, None)))

        # LLF
        llf += np.sum(np.log(np.take(1/denom, reference_category_indices.nonzero()[0])))

        ## LLF SATURATED (for deviance)
        #llf_saturated += np.sum(reference_category_indices * np.log(np.clip(reference_category_indices, 1e-10, None)))
        deviance = 2 * (llf_saturated - llf)

        return {
            'llf': llf,
            'deviance': deviance,
            'beta_update_data': results
        }

class OrdinalComputationUnit(ComputationUnit):
    @staticmethod
    def compute(data, y_label, X_labels, betas):
        X = data.to_pandas()[sorted(X_labels)]
        X['__const'] = 1
        X = X.to_numpy().astype(float)

        models = {}
        results = {}
        for level in betas.keys():
            level_int = int(level.split('__ord__')[-1])
            y = data.to_pandas()[y_label]
            y = (y.to_numpy() <= level_int).astype(float)

            models[level] = ComputationHelper.get_regression_model(
                y=y,
                X=X,
                beta=betas[level],
                glm_family=family.Binomial()
            )
            current_result = ComputationHelper.run_model(
                y=y,
                X=X,
                model=models[level]
            )
            results[level] = {'xwx': current_result['xwx'], 'xwz': current_result['xwz']}

        model_list = [(level, model.predict()) for level, model
                      in sorted(models.items(), key=lambda lvl: int(lvl[0].split('__ord__')[-1]))]

        llf = 0
        llf_saturated = 0
        # Calculate data for Y=1
        level, mu0 = model_list[0]
        mu0 = np.clip(mu0, 1e-10, 1-1e-10)
        level_int = int(level.split('__ord__')[-1])
        current_level_indices = data[y_label].to_numpy() == level_int
        reference_level_indices = 1-current_level_indices
        llf += np.sum(np.log(np.take(mu0, current_level_indices.nonzero()[0])))
        #llf_saturated += np.sum(current_level_indices * np.log(np.clip(current_level_indices, 1e-10, None)))

        # Running P(Y=k) = P(Y<=k) - P(Y<=k-1) for k=1...M-1
        for i in range(1,len(model_list)):
            _, mu0 = model_list[i-1]
            level, mu1 = model_list[i]

            mu0 = np.clip(mu0, 1e-10, 1-1e-10)
            mu1 = np.clip(mu1, 1e-10, 1-1e-10)

            mu_diff = np.clip(mu1 - mu0, 1e-10, 1-1e-10)

            # update reference category indices
            level_int = int(level.split('__ord__')[-1])
            current_level_indices = data[y_label].to_numpy() == level_int
            reference_level_indices = reference_level_indices * (1-current_level_indices)

            # LLF
            llf += np.sum(np.log(np.take(mu_diff, current_level_indices.nonzero()[0])))
            #llf_saturated += np.sum(current_level_indices * np.log(np.clip(current_level_indices, 1e-10, None)))
        # Calculate data for Y=M
        _, mu1 = model_list[-1]
        mu1 = np.clip(mu1, 1e-10, 1-1e-10)
        llf += np.sum(np.log(np.take(np.clip(1-mu1, 1e-10, 1-1e-10), reference_level_indices.nonzero()[0])))
        #llf_saturated += np.sum(reference_level_indices * np.log(np.clip(reference_level_indices, 1e-10, None)))
        deviance = 2 * (llf_saturated - llf)

        return {
            'llf': llf,
            'deviance': deviance,
            'beta_update_data': results
        }

polars_dtype_map = {
    pl.Float64: VariableType.CONTINUOS,
    pl.Boolean: VariableType.BINARY,
    pl.String: VariableType.CATEGORICAL,
    pl.Int32: VariableType.ORDINAL,
    pl.Int64: VariableType.ORDINAL
}

regression_computation_map = {
    VariableType.CONTINUOS: ContinousComputationUnit,
    VariableType.BINARY: BinaryComputationUnit,
    VariableType.CATEGORICAL: CategoricalComputationUnit,
    VariableType.ORDINAL: OrdinalComputationUnit
}

# TODO: move out __cat__, __ord__, and __const

class Client():
    def __init__(self, data: pl.DataFrame):
        self.data: pl.DataFrame = data
        self.schema: Dict[str, VariableType] = {column: polars_dtype_map[dtype] for column, dtype in dict(self.data.schema).items()}
        self.categorical_expressions: Dict[str, List[str]] = {column: self.data.select(column).to_dummies(separator='__cat__').columns
                                                              for column, dtype in self.schema.items() if dtype == VariableType.CATEGORICAL}
        self.ordinal_expressions: Dict[str, List[str]] = {column: self.data.select(pl.col(column)).to_dummies(separator='__ord__').columns
                                                          for column, dtype in self.schema.items() if dtype == VariableType.ORDINAL}

        self.server_categorical_expressions: Dict[str, List[str]] = None
        self.server_ordinal_expressions: Dict[str, List[str]] = None
        self.expanded_data: pl.DataFrame = None

    def get_data_schema(self):
        return self.schema

    def get_categorical_expressions(self):
        return self.categorical_expressions
    def get_ordinal_expressions(self):
        return self.ordinal_expressions

    def provide_expressions(
        self,
        categorical_expressions: Dict[str, List[str]],
        ordinal_expressions: Dict[str, List[str]]
    ):

        self.server_categorical_expressions = categorical_expressions
        self.server_ordinal_expressions = ordinal_expressions

        # expand categoricals
        all_possible_categorical_expressions = set([li for l in categorical_expressions.values() for li in l])
        _data = self.data.to_dummies([column for column, dtype in self.schema.items() if dtype == VariableType.CATEGORICAL], separator='__cat__')
        missing_cols = list(all_possible_categorical_expressions - set(_data.columns))
        _data = _data.with_columns(*[pl.lit(0.0).alias(c) for c in missing_cols])

        if EXPAND_ORDINALS == 1:
            # expand ordinals
            all_possible_ordinal_expressions = set([li for l in ordinal_expressions.values() for li in l])
            _data = _data.to_dummies([column for column, dtype in self.schema.items() if dtype == VariableType.ORDINAL], separator='__ord__')
            missing_cols = list(all_possible_ordinal_expressions - set(_data.columns))
            _data = _data.with_columns(*[pl.lit(0.0).alias(c) for c in missing_cols])

            # keep original ordinal variables
            _data = _data.with_columns(self.data.select([column for column, dtype in self.schema.items() if dtype == VariableType.ORDINAL]))

        self.expanded_data = _data

    def compute(self, y_label: str, X_labels, beta):
        assert y_label in self.schema

        result = regression_computation_map[self.schema[y_label]].compute(self.expanded_data, y_label, X_labels, beta)

        if self.schema[y_label] in [VariableType.CONTINUOS, VariableType.BINARY]:
            response: ClientResponseData = ClientResponseData(
                llf=result['llf'],
                deviance=result['deviance'],
                beta_update_data={list(beta.keys())[0]: BetaUpdateData(xwx=result['xwx'], xwz=result['xwz'])}
            )
        else:
            beta_update_data = {category: BetaUpdateData(xwx=data['xwx'], xwz=data['xwz']) for category, data in result['beta_update_data'].items()}
            response: ClientResponseData = ClientResponseData(
                llf=result['llf'],
                deviance=result['deviance'],
                beta_update_data=beta_update_data
            )

        return response