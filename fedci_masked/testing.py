from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy

from .env import DEBUG, FIT_INTERCEPT
from .utils import BetaUpdateData


class RegressionTest:
    def __init__(
        self,
        response: str,
        predictors: List[str],
        beta: np.ndarray,
        convergence_threshold: float = 1e-3,
        max_iterations: int = 25,
    ):
        self.response: str = response
        self.predictors: Set[str] = set(predictors)

        self.beta = beta
        self.dof = len(beta)

        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

        self.previous_llf = None
        self.llf: float = 0.0
        self.iterations: int = 0

    def __repr__(self):
        response, predictors, iteration, beta = self.get_test_parameters()
        predictors = sorted(list(predictors))
        return f"RegressionTest {response} ~ {', '.join(predictors + ['1'])} - iteration {iteration}/{self.max_iterations}{' - not finished' if not self.is_finished() else ''}"

    def get_change_in_llf(self):
        if self.previous_llf is None:
            return 1
        return abs(self.llf - self.previous_llf)

    def is_finished(self):
        return (
            self.get_change_in_llf() < self.convergence_threshold
            or self.iterations >= self.max_iterations
        )

    def get_test_parameters(self):
        return (
            self.response,
            tuple(sorted(list(self.predictors))),
            self.iterations,
            self.beta,
        )

    def update_parameters(self, update: List[BetaUpdateData]):
        if self.is_finished():
            return
        llf = sum([_update.llf for _update in update])
        xwx = sum([_update.xwx for _update in update])
        xwz = sum([_update.xwz for _update in update])

        if self.response == "A" and len(self.predictors) == 0:
            for i, _update in enumerate(update):
                print(i, _update.xwx)

        try:
            xwx_inv = np.linalg.inv(xwx)
        except np.linalg.LinAlgError:
            xwx_inv = np.linalg.pinv(xwx)

        if DEBUG > 3:
            print(
                f"{self.response} ~ {sorted(list(self.predictors))} - Iteration {self.iterations}/{self.max_iterations}"
            )
            print(
                f"{self.beta.reshape(-1).tolist()} -> {(xwx_inv @ xwz).reshape(-1).tolist()}"
            )
            print(
                f"{'None' if self.previous_llf is None else self.previous_llf} -> {self.llf}"
            )

        self.previous_llf = self.llf
        self.llf = llf

        self.beta = xwx_inv @ xwz

        self.iterations += 1


class LikelihoodRatioTest:
    def __init__(
        self, restricted_test: RegressionTest, unrestricted_test: RegressionTest
    ):
        assert restricted_test.response == unrestricted_test.response, (
            "The provided tests do not fit the same response variable"
        )
        self.response: str = restricted_test.response

        assert (
            len(set(restricted_test.predictors) - set(unrestricted_test.predictors))
            == 0
        ), "The provided tests are not properly nested"

        predictor_difference = list(
            set(unrestricted_test.predictors) - set(restricted_test.predictors)
        )

        assert len(predictor_difference) == 1, (
            "The provided tests differ in more than one variable"
        )

        self.test_variable: str = predictor_difference[0]
        self.conditioning_set: Set[str] = restricted_test.predictors

        self.restricted_test: RegressionTest = restricted_test
        self.unrestricted_test: RegressionTest = unrestricted_test

        self.p_value: Optional[float] = None

    def __repr__(self):
        if self.p_value is None:
            val = "not finished"
        else:
            val = f"p: {self.p_value:.4f}"
        restricted_test_string = self.restricted_test.__repr__()
        restricted_test_string = "\n\t".join(restricted_test_string.split("\n"))
        unrestricted_test_string = self.unrestricted_test.__repr__()
        unrestricted_test_string = "\n\t".join(unrestricted_test_string.split("\n"))
        return f"LikelihoodRatioTest - y: {self.response}, x: {self.test_variable}, S: {sorted(list(self.conditioning_set))}, {val}\n\t- {restricted_test_string}\n\t- {unrestricted_test_string}"

    def is_finished(self):
        finished = (
            self.restricted_test.is_finished() and self.unrestricted_test.is_finished()
        )
        if finished and self.p_value is None:
            self._set_p_value()
        return finished

    def get_test_parameters(self):
        test_parameters = {}
        if not self.restricted_test.is_finished():
            response, predictors, iteration, beta = (
                self.restricted_test.get_test_parameters()
            )
            test_parameters[(response, predictors, iteration)] = beta
        if not self.unrestricted_test.is_finished():
            response, predictors, iteration, beta = (
                self.unrestricted_test.get_test_parameters()
            )
            test_parameters[(response, predictors, iteration)] = beta
        return test_parameters

    def get_betas(self):
        betas = {}
        response, predictors, iteration, beta = (
            self.restricted_test.get_test_parameters()
        )
        betas[(response, predictors, iteration)] = beta
        response, predictors, iteration, beta = (
            self.unrestricted_test.get_test_parameters()
        )
        betas[(response, predictors, iteration)] = beta
        return betas

    def update_parameters(self, update: List[Dict[Tuple[str], BetaUpdateData]]):
        if not self.restricted_test.is_finished():
            self.restricted_test.update_parameters(
                [
                    _update[tuple(sorted(list(self.restricted_test.predictors)))]
                    for _update in update
                ]
            )
        if not self.unrestricted_test.is_finished():
            self.unrestricted_test.update_parameters(
                [
                    _update[tuple(sorted(list(self.unrestricted_test.predictors)))]
                    for _update in update
                ]
            )

    def _set_p_value(self):
        t0_llf = self.restricted_test.llf
        t1_llf = self.unrestricted_test.llf

        t0_dof = self.restricted_test.dof
        t1_dof = self.unrestricted_test.dof

        self.p_value = scipy.stats.chi2.sf(
            2 * (t1_llf - t0_llf), t1_dof - t0_dof
        ).item()

        if DEBUG >= 2:
            print(
                f"*** Calculating p value for independence of {self.response} from {self.test_variable} given {self.conditioning_set}"
            )
            print(f"{t1_dof - t0_dof} DOFs = {t1_dof} T1 DOFs - {t0_dof} T0 DOFs")
            print(
                f"{2 * (t1_llf - t0_llf):.4f} Test statistic = 2*({t1_llf:.4f} T1 LLF - {t0_llf:.4f} T0 LLF)"
            )
            print(f"p value = {self.p_value:.6f}")


class SymmetricLikelihoodRatioTest:
    def __init__(self, lrt1: LikelihoodRatioTest, lrt2: LikelihoodRatioTest):
        assert (
            lrt1.response == lrt2.test_variable
            and lrt1.test_variable == lrt2.response
            and lrt1.conditioning_set == lrt2.conditioning_set
        ), "The provided tests are not symmetrical"

        self.v0 = lrt1.response
        self.v1 = lrt2.response
        self.conditioning_set = lrt1.conditioning_set

        if lrt1.response < lrt2.response:
            self.lrt1: LikelihoodRatioTest = lrt1
            self.lrt2: LikelihoodRatioTest = lrt2
        else:
            self.lrt1: LikelihoodRatioTest = lrt2
            self.lrt2: LikelihoodRatioTest = lrt1

        self.p_value: Optional[float] = None

    def __repr__(self):
        if self.p_value is None:
            val = "not finished"
        else:
            val = f"p: {self.p_value:.4f}"
        lrt1_string = self.lrt1.__repr__()
        lrt1_string = "\n\t".join(lrt1_string.split("\n"))
        lrt2_string = self.lrt2.__repr__()
        lrt2_string = "\n\t".join(lrt2_string.split("\n"))
        return f"SymmetricLikelihoodRatioTest - v0: {self.v0}, v1: {self.v1}, conditioning set: {self.conditioning_set}, {val}\n\t- {lrt1_string}\n\t- {lrt2_string}"

    def is_finished(self):
        finished = self.lrt1.is_finished() and self.lrt2.is_finished()
        if finished and self.p_value is None:
            self._set_p_value()
        return finished

    def get_test_parameters(self):
        test_parameters = (
            self.lrt1.get_test_parameters() | self.lrt2.get_test_parameters()
        )
        return test_parameters

    def get_betas(self):
        betas = self.lrt1.get_betas() | self.lrt2.get_betas()
        return betas

    def update_parameters(
        self, update: List[Dict[str, Dict[Tuple[str], BetaUpdateData]]]
    ):
        if not self.lrt1.is_finished():
            self.lrt1.update_parameters(
                [_update[self.lrt1.response] for _update in update]
            )
        if not self.lrt2.is_finished():
            self.lrt2.update_parameters(
                [_update[self.lrt2.response] for _update in update]
            )

    def _set_p_value(self):
        self.p_value = min(
            2 * min(self.lrt1.p_value, self.lrt2.p_value),
            max(self.lrt1.p_value, self.lrt2.p_value),
        )

        if DEBUG >= 2:
            print(
                f"*** Combining p values for symmetry of tests between {self.lrt1.response} and {self.lrt2.response} given {self.lrt1.conditioning_set}"
            )
            print(f"p value {self.lrt1.response}: {self.lrt1.p_value}")
            print(f"p value {self.lrt2.response}: {self.lrt2.p_value}")
            print(f"p value = {self.p_value:.4f}")


class TestEngine:
    def __init__(
        self,
        schema,
        category_expressions,
        ordinal_expressions,
        convergence_threshold=1e-3,
        max_iterations=25,
    ):
        self.schema = schema
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.category_expressions = category_expressions
        self.ordinal_expressions = ordinal_expressions
        self.test = None

    def _get_number_of_parameters(self, resp_var, cond_vars):
        num_params = 1 if FIT_INTERCEPT else 0

        for var in cond_vars:
            if var in self.category_expressions:
                num_params += (
                    len(self.category_expressions[var]) - 1
                )  # -1 for ref category
            elif var in self.ordinal_expressions:
                num_params += (
                    len(self.ordinal_expressions[var]) - 1
                )  # -1 for ref category
            else:
                num_params += 1

        if resp_var in self.category_expressions:
            num_params *= (
                len(self.category_expressions[resp_var]) - 1
            )  # -1 for ref category
        elif resp_var in self.ordinal_expressions:
            num_params *= (
                len(self.ordinal_expressions[resp_var]) - 1
            )  # -1 for ref category
        return num_params

    def start_test(self, x: str, y: str, s: List[str]):
        if any([v not in self.schema for v in s + [x, y]]):
            raise Exception("The requested test requires unknown variables")

        y_s_params = self._get_number_of_parameters(y, s)
        y_sx_params = self._get_number_of_parameters(y, s + [x])
        x_s_params = self._get_number_of_parameters(x, s)
        x_sy_params = self._get_number_of_parameters(x, s + [y])

        test_y_restricted = RegressionTest(
            response=y,
            predictors=s,
            beta=np.zeros((y_s_params, 1)),
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )
        test_y_unrestricted = RegressionTest(
            response=y,
            predictors=s + [x],
            beta=np.zeros((y_sx_params, 1)),
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )
        test_x_restricted = RegressionTest(
            response=x,
            predictors=s,
            beta=np.zeros((x_s_params, 1)),
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )
        test_x_unrestricted = RegressionTest(
            response=x,
            predictors=s + [y],
            beta=np.zeros((x_sy_params, 1)),
            convergence_threshold=self.convergence_threshold,
            max_iterations=self.max_iterations,
        )

        lrt_y = LikelihoodRatioTest(
            restricted_test=test_y_restricted, unrestricted_test=test_y_unrestricted
        )

        lrt_x = LikelihoodRatioTest(
            restricted_test=test_x_restricted, unrestricted_test=test_x_unrestricted
        )

        self.test = SymmetricLikelihoodRatioTest(lrt1=lrt_x, lrt2=lrt_y)

    def is_finished(self):
        if self.test is None:
            return True
        return self.test.is_finished()

    def get_test_parameters(self):
        return self.test.get_test_parameters()

    def update_parameters(
        self, update: List[Dict[Tuple[str, Tuple[str], int], BetaUpdateData]]
    ):
        new_update = []
        for _update in update:
            _new_update = {self.test.v0: {}, self.test.v1: {}}
            for (resp_var, cond_vars, _), payload in _update.items():
                _new_update[resp_var][cond_vars] = payload
            new_update.append(_new_update)
        update: List[Dict[str, Dict[Tuple[str], BetaUpdateData]]] = new_update
        return self.test.update_parameters(update)

    def get_result(self):
        if not self.is_finished():
            return None
        return self.test
