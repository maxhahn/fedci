from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import rpyc
from scipy.special import softmax

from .env import ADDITIVE_MASKING, CLIENT_HETEROGENIETY, FIT_INTERCEPT
from .utils import (
    BetaUpdateData,
    VariableType,
    categorical_separator,
    constant_colname,
    ordinal_separator,
    polars_dtype_map,
)


class DistributionalFamily:
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def inverse_deriv(eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def variance(mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def loglik(y: np.ndarray, mu: np.ndarray) -> float:
        raise NotImplementedError()


class Gaussian(DistributionalFamily):
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return mu

    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return eta

    @staticmethod
    def inverse_deriv(eta: np.ndarray) -> np.ndarray:
        return np.ones_like(eta)

    @staticmethod
    def variance(mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    @staticmethod
    def loglik(y: np.ndarray, mu: np.ndarray) -> float:
        return -0.5 * np.sum((y - mu) ** 2)


class Binomial(DistributionalFamily):
    @staticmethod
    def link(mu: np.ndarray) -> np.ndarray:
        return np.log(mu / (1 - mu))

    @staticmethod
    def inverse_link(eta: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-eta))

    @staticmethod
    def inverse_deriv(eta: np.ndarray) -> np.ndarray:
        mu = Binomial.inverse_link(eta)
        return mu * (1 - mu)

    @staticmethod
    def variance(mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)

    @staticmethod
    def loglik(y: np.ndarray, mu: np.ndarray) -> float:
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        return np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))


class ComputationHelper:
    @staticmethod
    def run_regression(
        y: np.ndarray,
        X: np.ndarray,
        beta: np.ndarray,
        family: DistributionalFamily,
    ):
        eta: np.ndarray = X @ beta
        if CLIENT_HETEROGENIETY:
            gamma = ComputationHelper.fit_local_gamma(
                y=y, X=X, beta=beta, family=family
            )
            eta += gamma
        mu: np.ndarray = family.inverse_link(eta)

        dmu_deta: np.ndarray = family.inverse_deriv(eta)
        var_y: np.ndarray = family.variance(mu)

        W = np.diag(((dmu_deta**2) / var_y).reshape(-1))
        z: np.ndarray = eta + (y - mu) / dmu_deta

        xw = X.T @ W
        xwx = xw @ X
        xwz = xw @ z

        llf: float = family.loglik(y, mu)

        return {"llf": llf, "xwx": xwx, "xwz": xwz}

    @staticmethod
    def fit_local_gamma(y, X, beta, family, max_iter=20, tol=1e-8):
        gamma = 0.0
        offset = X @ beta

        for _ in range(max_iter):
            eta = gamma + offset

            mu = family.inverse_link(eta)
            dmu_deta = family.inverse_deriv(eta)
            var = family.variance(mu)

            # Score (first derivative)
            grad = np.sum((y - mu) * dmu_deta / var)

            # Fisher information (negative expected Hessian)
            w = (dmu_deta**2) / var
            fisher_info = np.sum(w)

            step = grad / fisher_info
            gamma = gamma + step

            if np.linalg.norm(step) < tol:
                break
        return gamma

    @staticmethod
    def fit_multinomial_gamma(y, X, beta, max_iter=20, tol=1e-8):
        eta_base = X @ beta
        n, K_minus_1 = y.shape
        K = K_minus_1 + 1

        # initialize gamma
        gamma = np.zeros(K_minus_1)

        for _ in range(max_iter):
            # eta including gamma, add reference category (0 column)
            eta = np.column_stack([eta_base + gamma, np.zeros(n)])
            p = softmax(eta, axis=1)

            # gradient: sum_i (Y_ik - p_ik)
            grad = np.sum(y - p[:, :K_minus_1], axis=0)

            # Hessian: H[a,b] = - sum_i p_ia * (1[a=b] - p_ib)
            H = np.zeros((K_minus_1, K_minus_1))
            for i in range(n):
                pi = p[i, :K_minus_1]
                for a in range(K_minus_1):
                    for b in range(K_minus_1):
                        H[a, b] -= pi[a] * ((1 if a == b else 0) - pi[b])

            # Newton step
            try:
                step = np.linalg.inv(H) @ grad
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H) @ grad
            gamma = gamma + step

            # convergence check
            if np.linalg.norm(step) < tol:
                break
        return gamma


def get_data(
    data: pl.DataFrame, response: str | List[str], predictors: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    if FIT_INTERCEPT:
        data = data.with_columns(pl.lit(1).alias(constant_colname))

    X: np.ndarray = data.select(predictors).to_numpy().astype(float)
    y: np.ndarray = data.select(response).to_numpy().astype(float)
    return (y, X)


class ComputationUnit:
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
    ):
        raise NotImplementedError()


class ContinousComputationUnit(ComputationUnit):
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
    ):
        if len(beta) == 0:
            print("Handle constant model")
        y, X = get_data(data, response, predictors)
        assert y.shape[0] == X.shape[0] and X.shape[1] == beta.shape[0], (
            "Shape mismatch between response, predictors, and beta"
        )
        return ComputationHelper.run_regression(y=y, X=X, beta=beta, family=Gaussian())


class BinaryComputationUnit(ComputationUnit):
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
    ):
        if len(beta) == 0:
            print("Handle constant model")
        y, X = get_data(data, response, predictors)
        assert y.shape[0] == X.shape[0] and X.shape[1] == beta.shape[0], (
            "Shape mismatch between response, predictors, and beta"
        )
        return ComputationHelper.run_regression(y=y, X=X, beta=beta, family=Binomial())


class CategoricalComputationUnit(ComputationUnit):
    @staticmethod
    def compute(
        data: pl.DataFrame,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
    ):
        # Identify the dummy columns for the response
        response_dummy_columns = [
            c
            for c in data.columns
            if c.startswith(f"{response}{categorical_separator}")
        ]

        y, X = get_data(data, response_dummy_columns, predictors)
        y = y[:, 1:]

        num_categories = len(response_dummy_columns)  # J
        num_features = len(predictors)  # K

        # Reshape beta (K x (J-1))
        beta = beta.reshape(num_features, -1, order="F")

        # gamma = np.zeros((len(y), beta.shape[1]))
        if CLIENT_HETEROGENIETY:
            gamma = ComputationHelper.fit_multinomial_gamma(y=y, X=X, beta=beta)
            gamma = np.tile(np.array(gamma), (y.shape[0], 1))
        else:
            gamma = np.zeros_like(y)

        # Compute eta and mu
        eta = np.clip(X @ beta + gamma, -350, 350)  # N x (J-1)
        mu = np.clip(
            softmax(np.column_stack([eta, np.zeros(y.shape[0])]), axis=1),
            1e-8,
            1 - 1e-8,
        )  # N x J
        mu_reduced = mu[:, 1:]  # N x (J-1)

        # Initialize accumulators for XWX and XWz
        XWX = np.zeros(
            (num_features * (num_categories - 1), num_features * (num_categories - 1))
        )
        XWz = np.zeros(num_features * (num_categories - 1))

        # Construct W blocks and z
        for i in range(y.shape[0]):
            y_i = y[i]  # (J-1)
            p_i = mu_reduced[i]
            var_i = np.diag(p_i) - np.outer(p_i, p_i)  # (J-1) x (J-1)

            try:
                var_i_inv = np.linalg.inv(var_i)
            except np.linalg.LinAlgError:
                var_i_inv = np.linalg.pinv(var_i)

            z_i = eta[i] + var_i_inv @ (y_i - p_i)  # (J-1)

            # Compute local contributions to XWX and XWz
            Xi = np.kron(np.eye(num_categories - 1), X[i : i + 1])  # (J-1) x (J-1)*K
            Wi = var_i  # (J-1) x (J-1)
            XWX += Xi.T @ Wi @ Xi
            XWz += Xi.T @ Wi @ z_i

        # Compute log-likelihood and deviance
        y_full = data.to_pandas()[response_dummy_columns].to_numpy()  # N x J
        logprob = np.log(np.clip(mu, 1e-8, 1))
        llf = np.sum(y_full * logprob)

        return {"llf": llf, "xwx": XWX, "xwz": XWz.reshape(-1, 1)}


regression_computation_map = {
    VariableType.CONTINUOS: ContinousComputationUnit,
    VariableType.BINARY: BinaryComputationUnit,
    VariableType.CATEGORICAL: CategoricalComputationUnit,
    VariableType.ORDINAL: None,  # OrdinalComputationUnit,
}


class Client:
    def __init__(
        self, id: str, data: pl.DataFrame, _network_fetch_function=lambda x: x
    ):
        self._network_fetch_function = _network_fetch_function
        self.id = id
        self.data: pl.DataFrame = data
        self.schema: Dict[str, VariableType] = {
            column: polars_dtype_map[dtype]
            for column, dtype in dict(self.data.schema).items()
        }

        for column in self.schema:
            assert categorical_separator not in column, (
                f"Variable name {column} contains reserved substring {categorical_separator}"
            )
            assert ordinal_separator not in column, (
                f"Variable name {column} contains reserved substring {ordinal_separator}"
            )
            assert constant_colname != column, (
                f"Variable name {column} is a reserved name"
            )

        self.categorical_expressions: Dict[str, List[str]] = {
            column: self.data.select(column)
            .to_dummies(separator=categorical_separator)
            .columns
            for column, dtype in self.schema.items()
            if dtype == VariableType.CATEGORICAL
        }
        self.ordinal_expressions: Dict[str, List[str]] = {
            column: self.data.select(pl.col(column))
            .to_dummies(separator=ordinal_separator)
            .columns
            for column, dtype in self.schema.items()
            if dtype == VariableType.ORDINAL
        }

        self.global_categorical_expressions: Optional[Dict[str, List[str]]] = None
        self.global_ordinal_expressions: Optional[Dict[str, List[str]]] = None
        self.expanded_data: Optional[pl.DataFrame] = None

        self.contributing_clients: Dict[str, Client] = {}
        self.received_masks = {}
        self.send_masks = {}
        self.response_masking = {}

    def get_id(self):
        return self.id

    def get_schema(self):
        return self.schema

    def get_categorical_expressions(self):
        return self.categorical_expressions

    def get_ordinal_expressions(self):
        return self.ordinal_expressions

    def set_clients(self, clients):
        del clients[self.id]  # remove self vom clients
        self.contributing_clients = clients

    def set_global_expressions(
        self,
        categorical_expressions: Dict[str, List[str]],
        ordinal_expressions: Dict[str, List[str]],
    ):
        self.global_categorical_expressions = categorical_expressions
        self.global_ordinal_expressions = ordinal_expressions

        # expand categoricals
        all_possible_categorical_expressions = set(
            [li for l in categorical_expressions.values() for li in l]
        )
        temp = self.data.select(
            [
                column
                for column, dtype in self.schema.items()
                if dtype == VariableType.CATEGORICAL
            ]
        )
        _data = self.data.to_dummies(
            [
                column
                for column, dtype in self.schema.items()
                if dtype == VariableType.CATEGORICAL
            ],
            separator=categorical_separator,
        )
        _data = _data.with_columns(temp)
        missing_cols = list(all_possible_categorical_expressions - set(_data.columns))
        _data = _data.with_columns(*[pl.lit(0.0).alias(c) for c in missing_cols])

        # expand ordinals
        all_possible_ordinal_expressions = set(
            [li for l in ordinal_expressions.values() for li in l]
        )
        _data = _data.to_dummies(
            [
                column
                for column, dtype in self.schema.items()
                if dtype == VariableType.ORDINAL
            ],
            separator=ordinal_separator,
        )
        missing_cols = list(all_possible_ordinal_expressions - set(_data.columns))
        _data = _data.with_columns(*[pl.lit(0.0).alias(c) for c in missing_cols])

        # keep original ordinal variables
        _data = _data.with_columns(
            self.data.select(
                [
                    column
                    for column, dtype in self.schema.items()
                    if dtype == VariableType.ORDINAL
                ]
            )
        )

        self.expanded_data = _data

    def exchange_masks(self, betas):
        for client_id, client in self.contributing_clients.items():
            for test_key, beta in betas.items():
                if test_key not in self.send_masks:
                    self.send_masks[test_key] = {}

                llf_mask = np.random.uniform(-1e8, 1e8, 1).item()
                xwx_mask = np.random.uniform(-1e8, 1e8, (len(beta), len(beta)))
                xwz_mask = np.random.uniform(-1e8, 1e8, (len(beta), 1))

                self.send_masks[test_key][client_id] = {
                    "llf": llf_mask,
                    "xwx": xwx_mask,
                    "xwz": xwz_mask,
                }
                client.add_mask(
                    self.id,
                    test_key,
                    llf_mask=llf_mask,
                    xwx_mask=xwx_mask,
                    xwz_mask=xwz_mask,
                )

    def add_mask(self, sender_id, test_key, llf_mask, xwx_mask, xwz_mask):
        assert sender_id in self.contributing_clients, "Unknown sender"
        if test_key not in self.received_masks:
            self.received_masks[test_key] = {}
        self.received_masks[test_key][sender_id] = {
            "llf": llf_mask,
            "xwx": xwx_mask,
            "xwz": xwz_mask,
        }

    def combine_masks(self, betas):
        if len(self.contributing_clients) == 0:
            return
        for test_key in betas:
            assert test_key in self.received_masks, (
                "Client did not receive required mask"
            )
            assert test_key in self.send_masks, "Client did not send required mask"
            assert len(self.received_masks[test_key]) == len(
                self.send_masks[test_key]
            ), "Number of send and received masks does not match"

            if test_key not in self.response_masking:
                self.response_masking[test_key] = {}

            llf_masks = []
            xwx_masks = []
            xwz_masks = []
            for sender_id, masks1 in self.received_masks[test_key].items():
                assert sender_id in self.send_masks[test_key], (
                    "Received a mask from a client no mask has been sent to"
                )

                masks2 = self.send_masks[test_key][sender_id]
                llf_masks.append(masks1["llf"] - masks2["llf"])
                xwx_masks.append(masks1["xwx"] - masks2["xwx"])
                xwz_masks.append(masks1["xwz"] - masks2["xwz"])

            self.response_masking[test_key] = {
                "llf": sum(llf_masks),
                "xwx": sum(xwx_masks),
                "xwz": sum(xwz_masks),
            }
            del self.received_masks[test_key]
            del self.send_masks[test_key]

    def apply_masks(self, test_key, irls_step_result):
        if not ADDITIVE_MASKING:
            return irls_step_result
        irls_step_result["llf"] += self.response_masking[test_key]["llf"]
        irls_step_result["xwx"] += self.response_masking[test_key]["xwx"]
        irls_step_result["xwz"] += self.response_masking[test_key]["xwz"]
        return irls_step_result

    def compute(
        self,
        betas: Dict[Tuple[str, Tuple[str], int], np.ndarray],
    ):
        betas = self._network_fetch_function(betas)

        test_vars = [
            {resp_var} | set(cond_vars) for resp_var, cond_vars, _ in betas.keys()
        ]
        test_vars = set.union(*test_vars)

        if any([v not in self.schema for v in test_vars]):
            result = {}
            for test_key, beta in betas.items():
                result[test_key] = self.apply_masks(
                    test_key,
                    {
                        "llf": 0,
                        "xwx": np.zeros((len(beta), len(beta))),
                        "xwz": np.zeros((len(beta), 1)),
                    },
                )
            return result

        results = {}
        for test_key, beta in betas.items():
            resp_var, cond_vars, _ = test_key

            new_cond_vars = []
            for cond_var in cond_vars:
                if cond_var in self.global_categorical_expressions:
                    new_cond_vars.extend(
                        self.global_categorical_expressions[cond_var][1:]
                    )
                elif cond_var in self.global_ordinal_expressions:
                    new_cond_vars.extend(self.global_ordinal_expressions[cond_var][1:])
                else:
                    new_cond_vars.append(cond_var)
            if FIT_INTERCEPT:
                new_cond_vars.append(constant_colname)
            cond_vars = new_cond_vars

            result = regression_computation_map[self.schema[resp_var]].compute(
                self.expanded_data, resp_var, cond_vars, beta
            )

            if len(self.contributing_clients) > 0:
                result = self.apply_masks(test_key, result)

            results[test_key] = BetaUpdateData(
                llf=result["llf"], xwx=result["xwx"], xwz=result["xwz"]
            )

        # TODO: maybe just exchange difference of LLF of full and nested models
        return results


class ProxyClient(rpyc.Service):
    def __init__(self, data):
        self.client = Client(data, _network_fetch_function=rpyc.classic.obtain)
        self.server: rpyc.utils.server.ThreadedServer = None

        # expose alle Methoden sofort:
        for name in dir(self.client):
            if callable(getattr(self.client, name)) and not name.startswith("_"):
                setattr(self, name, getattr(self.client, name))

    def __del__(self):
        self.close()

    def start(self, port):
        if self.server is not None:
            self.close()
        self.server = rpyc.utils.server.ThreadedServer(
            self,
            port=port,
            protocol_config={"allow_public_attrs": True, "allow_pickle": True},
        )
        self.server.start()

    def close(self):
        if self.server is None:
            return
        self.server.close()
        self.server = None
