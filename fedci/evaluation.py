from .testing import Test
from .env import DEBUG

import scipy
import numpy as np
import polars as pl
import polars.selectors as cs
import os
from typing import List

class LikelihoodRatioTest():
    def __init__(self, t0: Test, t1: Test) -> None:

        assert t0.y_label == t1.y_label, 'Provided tests do not predict the same variable'
        t0_req_labels = t0.get_required_labels() - {t0.y_label}
        t1_req_labels = t1.get_required_labels() - {t0.y_label}
        assert t0_req_labels.issubset(t1_req_labels), 'Provided tests are not nested'
        assert len(t0_req_labels)+1 == len(t1_req_labels), 'Provided tests differ by more than one regressor variable'

        self.y_label = t0.y_label
        self.x_label = list(t1_req_labels - t0_req_labels)[0]
        self.s_labels = sorted(list(t0_req_labels))

        self.p_val = self._run_ci_test(t0, t1)

    def _run_ci_test(self, t0: Test, t1: Test):
        client_subset = t1.get_providing_clients()
        t0_llf = t0.get_llf(client_subset)
        t1_llf = t1.get_llf(client_subset)

        t0_dof = t0.get_degrees_of_freedom()
        t1_dof = t1.get_degrees_of_freedom()

        p_val = scipy.stats.chi2.sf(2*(t1_llf - t0_llf), t1_dof-t0_dof)

        if DEBUG >= 2:
            print(f'*** Calculating p value for independence of {self.y_label} from {self.x_label} given {self.s_labels}')
            print(f'{t1_dof-t0_dof} DOFs = {t1_dof} T1 DOFs - {t0_dof} T0 DOFs')
            print(f'{2*(t1_llf - t0_llf):.4f} Test statistic = 2*({t1_llf:.4f} T1 LLF - {t0_llf:.4f} T0 LLF)')
            print(f'p value = {p_val:.6f}')
        return p_val

    def __repr__(self):
        return f"LikelihoodRatioTest - y: {self.y_label}, x: {self.x_label}, S: {self.s_labels}, p: {self.p_val:.4f}"

    def __lt__(self, other):
        if len(self.s_labels) < len(other.s_labels):
            return True
        elif len(self.s_labels) > len(other.s_labels):
            return False

        if self.y_label < other.y_label:
            return True
        elif self.y_label > other.y_label:
            return False

        if self.x_label < other.x_label:
            return True
        elif self.x_label > other.x_label:
            return False

        if tuple(sorted(self.s_labels)) < tuple(sorted(other.s_labels)):
            return True
        elif tuple(sorted(self.s_labels)) > tuple(sorted(other.s_labels)):
            return False

        return True

class SymmetricLikelihoodRatioTest():
    def __init__(self, lrt0: LikelihoodRatioTest, lrt1: LikelihoodRatioTest):

        assert lrt0.y_label == lrt1.x_label and lrt1.y_label == lrt0.x_label and sorted(lrt0.s_labels) == sorted(lrt1.s_labels), 'Tests do not match'

        self.lrt0: LikelihoodRatioTest = lrt0
        self.lrt1: LikelihoodRatioTest = lrt1

        self.v0, self.v1 = sorted([lrt0.y_label, lrt1.y_label])
        self.conditioning_set = sorted(lrt0.s_labels)

        self.p_val = min(2*min(self.lrt0.p_val, self.lrt1.p_val), max(self.lrt0.p_val, self.lrt1.p_val))
        if DEBUG >= 2:
            print(f'*** Combining p values for symmetry of tests between {self.v0} and {self.v1} given {self.conditioning_set}')
            print(f'p value {self.lrt0.y_label}: {self.lrt0.p_val}')
            print(f'p value {self.lrt1.y_label}: {self.lrt1.p_val}')
            print(f'p value = {self.p_val:.4f}')


    def __repr__(self):
        return f"SymmetricLikelihoodRatioTest - v0: {self.v0}, v1: {self.v1}, conditioning set: {self.conditioning_set}, p: {self.p_val:.4f}\n\t- {self.lrt0}\n\t- {self.lrt1}"

    def __lt__(self, other):
        if len(self.conditioning_set) < len(other.conditioning_set):
            return True
        elif len(self.conditioning_set) > len(other.conditioning_set):
            return False

        if self.v0 < other.v0:
            return True
        elif self.v0 > other.v0:
            return False

        if self.v1 < other.v1:
            return True
        elif self.v1 > other.v1:
            return False

        if tuple(self.conditioning_set) < tuple(other.conditioning_set):
            return True
        elif tuple(self.conditioning_set) > tuple(other.conditioning_set):
            return False

        return False

    def __eq__(self, other):
        return self.v0 == other.v0 and self.v1 == other.v1 and self.conditioning_set == other.conditioning_set

class EmptyLikelihoodRatioTest(SymmetricLikelihoodRatioTest):
    def __init__(self, v0, v1, conditioning_set, p_val):
        self.v0, self.v1 = sorted([v0, v1])
        self.conditioning_set = conditioning_set
        self.p_val = p_val

    def __repr__(self):
        return f"EmptyLikelihoodRatioTest - v0: {self.v0}, v1: {self.v1}, conditioning set: {self.conditioning_set}, p: {self.p_val:.4f}"

def get_likelihood_tests(tests: List[Test]):
    likelihood_tests = []
    for test in tests:
        curr_y = test.y_label
        curr_X = test.get_required_labels() - {curr_y}
        for x_var in curr_X:
            curr_conditioning_set = curr_X - {x_var}
            nested_test = [t for t in tests if ((t.get_required_labels() - {curr_y}) == curr_conditioning_set) and t.y_label == curr_y]
            if len(nested_test) == 0:
                print(f'No test for\n{test}\nin\n{tests}')
                continue
            assert len(nested_test) == 1, 'There is more than one nested test'
            likelihood_tests.append(LikelihoodRatioTest(nested_test[0], test))
    return likelihood_tests

def get_symmetric_likelihood_tests(tests, test_targets=None):
    symmetric_tests = []
    asymmetric_tests = get_likelihood_tests(tests)
    unique_tests = [t for t in asymmetric_tests if t.x_label < t.y_label]

    for test in unique_tests:
        if test_targets is not None and (test.x_label, test.y_label, tuple(sorted(test.s_labels))) not in test_targets:
            continue
        test_counterpart = [t for t in asymmetric_tests if (t.y_label == test.x_label) and (t.x_label == test.y_label) and (t.s_labels == test.s_labels)]
        if len(test_counterpart) == 0:
            continue
        assert len(test_counterpart) == 1, 'There is more than one matching counterpart test'
        symmetric_tests.append(SymmetricLikelihoodRatioTest(test, test_counterpart[0]))
    return symmetric_tests
