

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
        if type(model.family) == family.Gaussian:
            var_y = np.var(model.resid_response)
        elif type(model.family) == family.Binomial:
            var_y = dmu_deta
        else:
            raise Exception(f'Cannot handle model family {model.family.__class__.__name__}')
        W = np.diag((dmu_deta**2)/var_y)

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
    def compute(data, y_label, X_labels, beta):
        assert len(beta) == 1, 'Multinomial regression called with more than one beta'
        beta = beta[y_label]

        # Identify the dummy columns for the response
        y_dummy_columns = [c for c in data.columns if c.startswith(f'{y_label}__cat__')]

        # Design matrix
        X = data.to_pandas()[sorted(X_labels)]
        X['__const'] = 1
        X = X.to_numpy().astype(float)

        num_categories = len(y_dummy_columns)  # J
        num_features = len(X_labels) + 1       # K

        def softmax(eta):
            exp_eta = np.exp(np.hstack([np.zeros((eta.shape[0], 1)), eta]))
            return exp_eta / exp_eta.sum(axis=1, keepdims=True)

        # Response matrix (N x (J-1))
        Y = data.to_pandas()[y_dummy_columns[1:]].to_numpy()

        # Reshape beta (K x (J-1))
        beta = beta.reshape(num_features, -1, order='F')

        # Compute eta and mu
        eta = X @ beta               # N x (J-1)
        mu = softmax(eta)            # N x J
        mu_reduced = mu[:, 1:]       # N x (J-1)

        # Initialize accumulators for XWX and XWz
        XWX = np.zeros((num_features * (num_categories - 1), num_features * (num_categories - 1)))
        XWz = np.zeros(num_features * (num_categories - 1))

        # Construct W blocks and z
        for i in range(Y.shape[0]):
            yi = Y[i]  # (J-1)
            pi = mu_reduced[i]
            var_i = np.diag(pi) - np.outer(pi, pi)  # (J-1) x (J-1)

            try:
                var_i_inv = np.linalg.inv(var_i)
            except np.linalg.LinAlgError:
                var_i_inv = np.linalg.pinv(var_i)


            z_i = eta[i] + var_i_inv @ (yi - pi)  # (J-1)

            # Compute local contributions to XWX and XWz
            Xi = np.kron(np.eye(num_categories - 1), X[i:i+1])  # (J-1)*K x K
            Wi = var_i  # (J-1) x (J-1)
            XWX += Xi.T @ Wi @ Xi
            XWz += Xi.T @ Wi @ z_i

        # Compute log-likelihood and deviance
        Y_full = data.to_pandas()[y_dummy_columns].to_numpy()  # N x J
        logprob = np.log(np.clip(mu, 1e-8, 1))
        llf = np.sum(Y_full * logprob)
        deviance = -2 * llf

        results = {y_label: {'xwx': XWX, 'xwz': XWz}}

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
