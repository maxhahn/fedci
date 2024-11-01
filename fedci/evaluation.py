
from .testing import Test

import scipy
import polars as pl
import polars.selectors as cs
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
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
        
        self.p_val = round(self._run_ci_test(t0, t1), 4)
        
    def _run_ci_test(self, t0: Test, t1: Test):
        client_subset = t1.get_providing_clients()
        t0_llf = t0.get_llf(client_subset)
        t1_llf = t1.get_llf(client_subset)
        
        t0_dof = t0.get_degrees_of_freedom()
        t1_dof = t1.get_degrees_of_freedom()
        
        p_val = scipy.stats.chi2.sf(2*(t1_llf - t0_llf), t1_dof-t0_dof)
        return p_val
        
    def __repr__(self):
        return f"LikelihoodRatioTest - y: {self.y_label}, x: {self.x_label}, S: {self.s_labels}, p: {self.p_val}"
    
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
        
    def __repr__(self):
        return f"SymmetricLikelihoodRatioTest - v0: {self.v0}, v1: {self.v1}, conditioning set: {self.conditioning_set}, p: {self.p_val}\n\t- {self.lrt0}\n\t- {self.lrt1}"
    
    def __lt__(self, other):
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
        
        return True
    
    def __eq__(self, other):
        return self.v0 == other.v0 and self.v1 == other.v1 and self.conditioning_set == other.conditioning_set
    
class EmptyLikelihoodRatioTest(SymmetricLikelihoodRatioTest):
    def __init__(self, v0, v1, conditioning_set, p_val):
        self.v0, self.v1 = sorted([v0, v1])
        self.conditioning_set = conditioning_set
        self.p_val = round(p_val, 4)

    def __repr__(self):
        return f"EmptyLikelihoodRatioTest - v0: {self.v0}, v1: {self.v1}, conditioning set: {self.conditioning_set}, p: {self.p_val}"

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

def get_symmetric_likelihood_tests(tests):
    symmetric_tests = []
    asymmetric_tests = get_likelihood_tests(tests)
    unique_tests = [t for t in asymmetric_tests if t.y_label < t.x_label]

    for test in unique_tests:
        test_counterpart = [t for t in asymmetric_tests if (t.y_label == test.x_label) and (t.x_label == test.y_label) and (t.s_labels == test.s_labels)]
        if len(test_counterpart) == 0:
            continue
        assert len(test_counterpart) == 1, 'There is more than one matching counterpart test'
        symmetric_tests.append(SymmetricLikelihoodRatioTest(test, test_counterpart[0]))
    return symmetric_tests

def _transform_dataframe(df):
    pandas2ri.activate()
    base = importr('base')
    df_copy = df.copy()
    r_list = {}#ro.ListVector({})
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'float64':
            r_list[col] = pandas2ri.py2rpy(df_copy[col])
        elif df_copy[col].dtype == 'object':
            r_list[col] = base.as_factor(pandas2ri.py2rpy(df_copy[col]))
        elif df_copy[col].dtype == 'int64':
            unique_values = sorted(df_copy[col].unique())
            r_list[col] = base.factor(pandas2ri.py2rpy(df_copy[col]), 
                                    levels=ro.IntVector(unique_values), 
                                    ordered=True)
        else:
            raise Exception(f'Could not properly convert column {col} with type {df_copy[col].dtype}')
            
    r_list = ro.ListVector(r_list)
    r_dataframe = base.as_data_frame(r_list)
    return r_dataframe

def get_riod_tests(data, max_regressors=None):
    if max_regressors is None:
        max_regressors = 999
        
    data = data.with_columns(cs.integer().cast(pl.Int64))
    ground_truth_tests = []
        
    with (ro.default_converter + pandas2ri.converter).context():
        ro.r['source']('./local-ci.r')
        run_ci_test_f = ro.globalenv['run_ci_test']
        df_r = _transform_dataframe(data.with_columns(cs.boolean().cast(pl.Utf8)).to_pandas())
        
        # delete potentially existing file
        if os.path.exists('./tmp/citestResults_dummy.csv'):
            os.remove('./tmp/citestResults_dummy.csv')
        result = run_ci_test_f(df_r, max_regressors, "./tmp/", 'dummy')
        df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
        labels = list(result['labels'])
        
    df = pl.from_pandas(df_pvals)
    df = df.drop('ord')
    df = df.with_columns(pl.col('S').str.split(',').cast(pl.List(pl.Int64)))
    df = df.with_columns(pl.col('X', 'Y').cast(pl.Int64))
    
    for row in df.rows():
        x = labels[row[0]-1]
        y = labels[row[1]-1]
        if x > y:
            x,y = y,x
        s = [labels[r-1] for r in row[2] if r is not None]
        pval = row[3]
        ground_truth_tests.append(EmptyLikelihoodRatioTest(x, y, s, pval))
        
    return ground_truth_tests
    
def compare_tests_to_truth(tests: List[SymmetricLikelihoodRatioTest], ground_truth_tests: List[SymmetricLikelihoodRatioTest]):
    assert len(tests) == len(ground_truth_tests), f'Number of tests do not match: {len(tests)} != {len(ground_truth_tests)}'
    
    p_values = []
    for test, gt_test in zip(sorted(tests), sorted(ground_truth_tests)):
        if test != gt_test:
            print(test)
            print(gt_test)
            raise Exception('Mismatched tests in sorted zip. This should not happen when all tests are present.')
        p_values.append((test.p_val, gt_test.p_val))
    return [p_vals[0] for p_vals in p_values], [p_vals[1] for p_vals in p_values]

    
    