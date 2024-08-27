import polars as pl
import polars.selectors as cs
import numpy as np
import math
import pickle
from typing import List
from itertools import chain, combinations
import matplotlib.pyplot as plt
from collections import defaultdict

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.genmod.families import family

import graphviz
import networkx as nx
from cdt.data import AcyclicGraphGenerator
import scipy

from scipy import stats
from pycit import citest
from pgmpy.estimators import CITests
from tqdm import tqdm

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from collections import OrderedDict


# Testing Classes

class TestingRound:
    iterations = 0
    last_deviance = None
    deviance = 0
    convergence_threshold = 1e-8
    rss = None
    llf = None
    total_samples = None
    
    def __init__(self, y_label, X_labels, tikhonov_lambda=0):
        self.y_label = y_label
        self.X_labels = X_labels
        self.tikhonov_lambda = tikhonov_lambda
        self._init_beta0()
        self.client_data = {
            'xtx': [],
            'xtz': [],
            'dev': [],
            'llf': [],
            'rss': [],
            'nobs': []
            }
        
    def __eq__(self, t):
        assert type(t) == TestingRound
        return self.y_label == t.y_label and self.X_labels == t.X_labels   
    
    def __repr__(self):
        return f'TestingRound - y: {self.y_label}, X: {self.X_labels}, total samples: {self.total_samples}, beta: {self.beta}, current iteration: {self.iterations}, current deviance: {abs(self.deviance)}, relative deviance change: {abs(self.deviance - self.last_deviance) / (0.1 + abs(self.deviance)) if self.last_deviance is not None else "?"}, llf: {self.llf}, rss: {self.rss}' 
    
    def _init_beta0(self):
        #self.beta = np.random.randn(len(self.X_labels) + 1) # +1 for intercept
        self.beta = np.zeros(len(self.X_labels) + 1) # +1 for intercept
        
    def get_relative_change_in_deviance(self):
        if self.last_deviance is None:
            return None
        return abs(self.deviance - self.last_deviance) / (0.1 + abs(self.deviance))
    
    def get_fit_stats(self, client_subset=None):
        #dev = [v for k,v in self.client_data['dev'].items() if client_subset is None or k in client_subset]
        llf = [v for k,v in self.client_data['llf'].items() if client_subset is None or k in client_subset]
        rss = [v for k,v in self.client_data['rss'].items() if client_subset is None or k in client_subset]
        nobs = [v for k,v in self.client_data['nobs'].items() if client_subset is None or k in client_subset]
    
        return {'llf': sum(llf), 'rss': sum(rss), 'nobs': sum(nobs)}
    
    def get_test_parameters(self):
        return self.y_label, self.X_labels, self.beta
    
    def update_llf(self, results):
        self.providing_clients = set(results.keys())
        self.llf = sum(results.values())
    
    def update_state(self, results):
        self.providing_clients = set(results.keys())
        
        client_data = [[(k,vi) for vi in v] for k,v in results.items()]
        xtx, xtz, dev, llf, rss, nobs = zip(*client_data)
        xtx, xtz, dev, llf, rss, nobs = dict(xtx), dict(xtz), dict(dev), dict(llf), dict(rss), dict(nobs)
        
        self.client_data['xtx'] = xtx
        self.client_data['xtz'] = xtz
        self.client_data['dev'] = dev
        self.client_data['llf'] = llf
        self.client_data['rss'] = rss
        self.client_data['nobs'] = nobs
        return
        
    def aggregate_results(self, results):
        self.update_state(results)
        
        xtx = self.client_data['xtx']  
        xtz = self.client_data['xtz']  
        dev = self.client_data['dev']  
        llf = self.client_data['llf']  
        rss = self.client_data['rss']  
        nobs = self.client_data['nobs']
        
        xtx_agg = sum(xtx.values())
        xtx_agg = xtx_agg + self.tikhonov_lambda*np.eye(xtx_agg.shape[0]) # tikhonov/ridge regularization
        
        self.beta = np.linalg.inv(xtx_agg) @ sum(xtz.values())
        self.last_deviance = self.deviance
        self.deviance = sum(dev.values())
        self.llf = sum(llf.values())
        self.rss = sum(rss.values())
        self.total_samples = sum(nobs.values())
        self.iterations += 1
        
        return self.get_relative_change_in_deviance() < self.convergence_threshold
    
class TestingEngine:
    def __init__(self, available_data, category_expressions, ordinal_expressions, tikhonov_lambda=0, max_regressors=None, max_iterations=25, save_steps=10):
        self.available_data = available_data
        self.max_regressors = max_regressors
        self.max_iterations = max_iterations
        self.save_steps = save_steps
        
        self.tikhonov_lambda = tikhonov_lambda
        
        self.finished_rounds = []
        self.testing_rounds = []
        
        self.category_expressions = category_expressions
        self.ordinal_expressions = ordinal_expressions
        
       # _max_conditioning_set_size = min(len(self.available_data)-1, self.max_regressors) if self.max_regressors is not None else len(self.available_data)-1
        
        for y_var in available_data:
            set_of_regressors = available_data - {y_var}
                
            _max_conditioning_set_size = min(len(set_of_regressors), self.max_regressors) if self.max_regressors is not None else len(set_of_regressors)
            powerset_of_regressors = chain.from_iterable(combinations(set_of_regressors, r) for r in range(0,_max_conditioning_set_size+1))
            
            # expand categorical features in regressor sets
            temp_powerset = []
            for var_set in powerset_of_regressors:
                for category, expressions in category_expressions.items():
                    if category in var_set:
                        var_set = (set(var_set) - {category}) | set(sorted(list(expressions)[1:])) # [1:] to drop first cat
                #for category, expressions in ordinal_expressions.items():
                #    if category in var_set:
                #        var_set = (set(var_set) - {category}) | set(sorted(list(expressions)[:-1])) # [1:] to drop first cat
                temp_powerset.append(var_set)
            powerset_of_regressors = temp_powerset
            
            if y_var in category_expressions:
                for y_var_cat in category_expressions[y_var]:
                    self.testing_rounds.extend([TestingRound(y_label=y_var_cat, X_labels=sorted(list(x_vars)), tikhonov_lambda=self.tikhonov_lambda) for x_vars in powerset_of_regressors])
            elif y_var in ordinal_expressions:
                for y_var_ord in ordinal_expressions[y_var][:-1]: # skip last category of ordinal regression
                    self.testing_rounds.extend([TestingRound(y_label=y_var_ord, X_labels=sorted(list(x_vars)), tikhonov_lambda=self.tikhonov_lambda) for x_vars in powerset_of_regressors])
            else:
                self.testing_rounds.extend([TestingRound(y_label=y_var, X_labels=sorted(list(x_vars)), tikhonov_lambda=self.tikhonov_lambda) for x_vars in powerset_of_regressors])
            
        self.testing_rounds = sorted(self.testing_rounds, key=lambda key: len(key.X_labels))
        self.is_finished = len(self.testing_rounds) == 0
        
    def get_finished_tests_by_y_label(self, y_labels):
        res = [t for t in self.finished_rounds if t.y_label in y_labels]
        return res
        
    def get_current_test_parameters(self):
        return self.testing_rounds[0].get_test_parameters()
    
    def remove_current_test(self):
        self.testing_rounds.pop(0)
        self.is_finished = len(self.testing_rounds) == 0
    
    def finish_current_test(self):
        self.finished_rounds.append(self.testing_rounds.pop(0))
        self.is_finished = len(self.testing_rounds) == 0
        
    def aggregate_results(self, results):
        has_converged = self.testing_rounds[0].aggregate_results(results)
        has_reached_max_iterations = self.testing_rounds[0].iterations >= self.max_iterations
        if has_converged or has_reached_max_iterations:
            self.finish_current_test()
            

class Server:
    def __init__(self, clients, tikhonov_lambda=0, max_regressors=None):
        self.clients = clients
        self.available_data = set.union(*[set(c.data_labels) for c in self.clients.values()])
        self.category_expressions = {}
        self.ordinal_expressions = {}
        for _, client in self.clients.items():
            for feature, expressions in client.get_categories().items():
                self.category_expressions[feature] = sorted(list(set(self.category_expressions.get(feature, [])).union(set(expressions))))
            for feature, expressions in client.get_ordinals().items():
                self.ordinal_expressions[feature] = sorted(list(set(self.ordinal_expressions.get(feature, [])).union(set(expressions))))
        self.reversed_category_expressions = {vi:k for k,v in self.category_expressions.items() for vi in v}
        self.reversed_ordinal_expressions = {vi:k for k,v in self.ordinal_expressions.items() for vi in v}
        
        for client in self.clients.values(): client.receive_category_expressions(self.category_expressions)

        self.testing_engine = TestingEngine(self.available_data,
                                            self.category_expressions,
                                            self.ordinal_expressions,
                                            tikhonov_lambda=tikhonov_lambda,
                                            max_regressors=max_regressors)
        
    def run_tests(self):
        counter = 1
        while not self.testing_engine.is_finished:
            y_label, X_labels, beta = self.testing_engine.get_current_test_parameters()
            
            client_labels = []
            #print(y_label, X_labels)
            for label in [y_label] + X_labels:
                if label in self.reversed_category_expressions:
                    #client_labels.append(self.reversed_category_expressions[label])
                    client_labels.append(label)
                elif label in self.reversed_ordinal_expressions:
                    client_labels.append(self.reversed_ordinal_expressions[label])
                else:
                    client_labels.append(label)                    
            
            selected_clients = {id_: c for id_, c in self.clients.items() if set(client_labels).issubset(c.extended_data_labels)}
            if len(selected_clients) == 0:
                print('WARNING! No client fulfills data requirements, removing current test...')
                self.testing_engine.remove_current_test()
                continue
            # http response, to compute glm results for y regressed on X with beta
            results = {id_:c.compute(y_label, X_labels, beta) for id_,c in selected_clients.items()}
            self.testing_engine.aggregate_results(results)
            if counter % self.testing_engine.save_steps == 0:
                counter = 0
                #with open('./testengine.ckp', 'wb') as f:
                #    pickle.dump(self.testing_engine, f)
            counter += 1
            
        # fix categorical tests
        self.update_categorical_tests()
        # fix ordinal llfs
        self.update_ordinal_tests()
        
    def update_categorical_tests(self):
        def _get_llfs(t0, t_ref):            
            client_labels = []            
            for label in [t0.y_label] + t0.X_labels:
                if label in self.reversed_category_expressions:
                    #client_labels.append(self.reversed_category_expressions[label])
                    client_labels.append(label)
                elif label in self.reversed_ordinal_expressions:
                    client_labels.append(self.reversed_ordinal_expressions[label])
                else:
                    client_labels.append(label)                    
            selected_clients = {id_: c for id_, c in self.clients.items() if set(client_labels).issubset(c.extended_data_labels)}
            
            assert len(selected_clients) > 0
            
            results = {id_:c.compute_categorical_llf(t0.X_labels,
                                                t0.y_label,
                                                t_ref.y_label,
                                                t0.beta,
                                                t_ref.beta
                                                ) for id_,c in selected_clients.items()}
            return results
        
        for categorical_key, categorical_values in self.category_expressions.items():
            all_categorical_tests = self.testing_engine.get_finished_tests_by_y_label(categorical_values)
            categorical_test_x_label_mapping = {}
            for t in all_categorical_tests:
                x_label_key = tuple(sorted(list(t.X_labels)))
                if x_label_key not in categorical_test_x_label_mapping:
                    categorical_test_x_label_mapping[x_label_key] = []
                categorical_test_x_label_mapping[x_label_key].append(t)
                
            for x_label_key, categorical_tests in categorical_test_x_label_mapping.items():
                #print('LLFs Before', [t.llf for t in categorical_tests])
                for i in range(len(categorical_tests)-1):    
                    results = _get_llfs(categorical_tests[i], categorical_tests[-1])
                    
                    #print(results)
                    #print('Updating', y_label, X_labels)
                    #print('Old llf', categorical_tests[i].llf)
                    categorical_tests[i].update_llf(results)
                    #print('New llf', categorical_tests[i].llf)
                #results = _get_llfs(categorical_tests[-1], None)
                #categorical_tests[-1].update_llf(results)
                #print('LLFs After', [t.llf for t in categorical_tests])
            
    def update_ordinal_tests(self):
        def _get_llfs(t0, t1):
            if t1 is None:
                t1_y = None
                t1_b = None
                required_labels = [t0.y_label] + t0.X_labels
            else:
                t1_y = t1.y_label
                t1_b = t1.beta
                required_labels = [t1.y_label] + t1.X_labels
            
            client_labels = []            
            for label in required_labels:
                if label in self.reversed_category_expressions:
                    #client_labels.append(self.reversed_category_expressions[label])
                    client_labels.append(label)
                elif label in self.reversed_ordinal_expressions:
                    client_labels.append(self.reversed_ordinal_expressions[label])
                else:
                    client_labels.append(label)                    
            selected_clients = {id_: c for id_, c in self.clients.items() if set(client_labels).issubset(c.extended_data_labels)}
            
            assert len(selected_clients) > 0
            
            results = {id_:c.compute_ordinal_llf(t0.X_labels,
                                                t0.y_label,
                                                t1_y,
                                                t0.beta,
                                                t1_b
                                                ) for id_,c in selected_clients.items()}
            return results
        
        for ordinal_key, ordinal_values in self.ordinal_expressions.items():
            all_ordinal_tests = self.testing_engine.get_finished_tests_by_y_label(ordinal_values)
            ordinal_test_x_label_mapping = {}
            for t in all_ordinal_tests:
                x_label_key = tuple(sorted(list(t.X_labels)))
                if x_label_key not in ordinal_test_x_label_mapping:
                    ordinal_test_x_label_mapping[x_label_key] = []
                ordinal_test_x_label_mapping[x_label_key].append(t)
                
            for x_label_key, ordinal_tests in ordinal_test_x_label_mapping.items():
                
                ordinal_tests = sorted(ordinal_tests, key=lambda x: int(x.y_label.split('__ord__')[-1]))
                #print('LLFs Before', [t.llf for t in ordinal_tests])
                for i in range(1,len(ordinal_tests)):    
                    results = _get_llfs(ordinal_tests[i-1], ordinal_tests[i])
                    
                    #print(results)
                    #print('Updating', y_label, X_labels)
                    #print('Old llf', ordinal_tests[i].llf)
                    ordinal_tests[i].update_llf(results)
                    #print('New llf', ordinal_tests[i].llf)
                results = _get_llfs(ordinal_tests[-1], None)
                ordinal_tests[-1].update_llf(results)
                #print('LLFs After', [t.llf for t in ordinal_tests])
 
# TODO: cat to binaries in order to perform ordinal regression               
#def category_to_binaries(df, cols):
#    for col in cols:
#        n_unique = df[col].n_unique()
#        for i in range(n_unique):
#            
#    return df

import enum

class VariableType(enum.Enum):
    CONTINUOS = 0
    CATEGORICAL = 1
    ORDINAL = 2
    
class Client:
    def __init__(self, data):
        self.data = data#.to_dummies(cs.string(), separator='__cat__', drop_first=True).cast(pl.Float64)
        self.data_labels = sorted(self.data.columns)
        self.extended_data_labels = sorted(self.data.to_dummies(cs.string(), separator='__cat__').columns)
        self.schema = {}
        for k,v in dict(self.data.schema).items():
            if v == pl.Float64:
                var_type = VariableType.CONTINUOS
            elif v == pl.String:
                var_type = VariableType.CATEGORICAL
            elif v == pl.Int32:
                var_type = VariableType.ORDINAL
            else:
                raise Exception('Unknown schema type encountered')
            
            self.schema[k] = var_type
            
        self.tikhonov_lambda = 1e-5
            
    def receive_category_expressions(self, expressions):
        self.category_expressions = set([li for l in expressions.values() for li in l])
        
    def filter_categoricals(self, column, values):
        if column not in self.schema or self.schema[column] != VariableType.CATEGORICAL:
            return
        self.data = self.data.filter(pl.col(column).is_in(values))
    
    def get_categories(self):
        result = {}
        for column, var_type in self.schema.items():
            if var_type != VariableType.CATEGORICAL:
                continue
            result[column] = self.data[column].to_dummies(separator='__cat__').columns
        return result

    def get_ordinals(self):
        result = {}
        for column, var_type in self.schema.items():
            if var_type != VariableType.ORDINAL:
                continue
            result[column] = [f"{column}__ord__{c}" for c in sorted(self.data[column].unique())]
            #result[column] = sorted(self.data[column].unique())
        return result
    
    def compute_categorical_llf(self, X_labels, y_label, y_label_ref_cat, beta, beta_ref_cat):
        def _get_probas(y_label: str, X_labels: List[str], beta):
            _data = self.data
            y_label, cat = y_label.split("__cat__")
            _data = _data.with_columns(pl.when(pl.col(y_label) == cat)
                                       .then(pl.lit(1.0))
                                       .otherwise(pl.lit(0.0))
                                       .alias(y_label))
            _data = _data.to_dummies(cs.string(), separator='__cat__').cast(pl.Float64)
            missing_cols = list(self.category_expressions - set(_data.columns))
            _data = _data.with_columns(*[pl.lit(0).alias(c) for c in missing_cols])
  
            X = _data.to_pandas()[X_labels]
            X = X.to_numpy().astype(float)
            X = sm.tools.add_constant(X) 
            
            y = _data.to_pandas()[y_label]
            y = y.to_numpy().astype(float)
            
            glm_model = sm.GLM(y, X, family=family.Binomial())
            glm_results = GLMResults(glm_model, beta, normalized_cov_params=None, scale=None)
            proba = glm_results.predict()
            
            proba = np.abs(np.abs((1-y)) - proba)
            
            return proba
            
        def _get_eta(y_label: str, X_labels: List[str], beta):
            _data = self.data
            y_label, cat = y_label.split("__cat__")
            _data = _data.with_columns(pl.when(pl.col(y_label) == cat)
                                       .then(pl.lit(1.0))
                                       .otherwise(pl.lit(0.0))
                                       .alias(y_label))
            _data = _data.to_dummies(cs.string(), separator='__cat__').cast(pl.Float64)
            missing_cols = list(self.category_expressions - set(_data.columns))
            _data = _data.with_columns(*[pl.lit(0).alias(c) for c in missing_cols])
  
            X = _data.to_pandas()[X_labels]
            X = X.to_numpy().astype(float)
            X = sm.tools.add_constant(X) 
            
            y = _data.to_pandas()[y_label]
            y = y.to_numpy().astype(float)
            
            glm_model = sm.GLM(y, X, family=family.Binomial())
            glm_results = GLMResults(glm_model, beta, normalized_cov_params=None, scale=None)
            eta = glm_results.predict(which='linear')
            
            return eta
        
        probas = _get_probas(y_label_ref_cat, X_labels, beta_ref_cat)
        eta = _get_eta(y_label, X_labels, beta)

        probas = probas * np.exp(eta)
        
        y_label, cat = y_label.split("__cat__")
        _data = self.data
        cat_association = _data.select(pl.when(pl.col(y_label) == cat)
                                .then(pl.lit(1.0))
                                .otherwise(pl.lit(0.0))
                                .alias(y_label))[y_label].to_numpy().astype(float)
        cat_association = np.where(cat_association == 1)[0]
        
        #probas = probas1 - probas0
        llf = np.sum(np.log(np.take(probas, cat_association)))
        
        return llf
    
    def compute_ordinal_llf(self, X_labels, y_label0, y_label1, beta0, beta1):
        def _get_probas(y_label: str, X_labels: List[str], beta):
            _data = self.data
            _data = _data.to_dummies(cs.string(), separator='__cat__').cast(pl.Float64)
            missing_cols = list(self.category_expressions - set(_data.columns))
            _data = _data.with_columns(*[pl.lit(0).alias(c) for c in missing_cols])

            y_label, cutoff = y_label.split('__ord__')
            _data = _data.with_columns(pl.when(pl.col(y_label) <= int(cutoff))
                                    .then(pl.lit(1.0))
                                    .otherwise(pl.lit(0.0))
                                    .alias(y_label))
                
            X = _data.to_pandas()[X_labels]
            X = X.to_numpy().astype(float)
            X = sm.tools.add_constant(X) 
            
            y = _data.to_pandas()[y_label]
            y = y.to_numpy().astype(float)
            
            glm_model = sm.GLM(y, X, family=family.Binomial())
            glm_results = GLMResults(glm_model, beta, normalized_cov_params=None, scale=None)
            proba = glm_results.predict()
            
            proba = np.abs(np.abs((1-y)) - proba)
            
            return proba, glm_results.llf
        
        probas0, llf0 = _get_probas(y_label0, X_labels, beta0)
        if y_label1 is None or beta1 is None:
            probas1, llf1 = np.ones(probas0.shape), None
        else:
            probas1, llf1 = _get_probas(y_label1, X_labels, beta1)
            
        y_label, cat = y_label0.split('__ord__')
            
        _data = self.data
        cat_association = _data.select(pl.when(pl.col(y_label) == int(cat))
                                .then(pl.lit(1.0))
                                .otherwise(pl.lit(0.0))
                                .alias(y_label)).to_pandas()[y_label].to_numpy().astype(float)
        cat_association = np.where(cat_association == 1)[0]
        
        probas = probas1 - probas0
        llf = np.sum(np.log(np.take(probas, cat_association)))
        
        return llf
    
    # def compute_category_restricted(self, y_label, X_labels, beta):
    #     _data = self.data.to_dummies(cs.string(), separator='__cat__').cast(pl.Float64)
    #     _data = _data.filter(pl.col(y_label) == 1)
    #     return self._compute(_data, y_label, X_labels, beta)
        
    def compute(self, y_label: str, X_labels: List[str], beta):  
        _data = self.data
        _data = _data.to_dummies(cs.string(), separator='__cat__').cast(pl.Float64)
        missing_cols = list(self.category_expressions - set(_data.columns))
        _data = _data.with_columns(*[pl.lit(0).alias(c) for c in missing_cols])
        if '__ord__' in y_label:
            y_label, cutoff = y_label.split('__ord__')
            _data = _data.with_columns(pl.when(pl.col(y_label) <= int(cutoff))
                                    .then(pl.lit(1.0))
                                    .otherwise(pl.lit(0.0))
                                    .alias(y_label))
        return self._compute(_data, y_label, X_labels, beta)
        
    def _compute(self, data, y_label: str, X_labels: List[str], beta):
        #print(y_label, X_labels)
        X = data.to_pandas()[X_labels]
        X = X.to_numpy().astype(float)
        X = sm.tools.add_constant(X) 
        
        y = data.to_pandas()[y_label]
        y = y.to_numpy().astype(float)
        
        do_log_reg = y_label in self.get_categories() or (y_label in self.schema and self.schema[y_label] == VariableType.ORDINAL)
        
        result = self._run_regression(y,X,beta,do_log_reg)
        
        return result
        
        
    def _run_regression(self, y, X, beta, do_log_reg):
        if do_log_reg:
            glm_model = sm.GLM(y, X, family=family.Binomial())
        else:
            glm_model = sm.GLM(y, X, family=family.Gaussian())
        #normalized_cov_params = np.linalg.inv(X.T.dot(X)) # singular matrix problem with missing cat_1 in data slices
        glm_results = GLMResults(glm_model, beta, normalized_cov_params=None, scale=None)
        
        #print(beta)
        #print(beta.shape)
        #print(X.shape)
        eta = glm_results.predict(which='linear')
        
        # g' is inverse of link function
        inverse_link = glm_results.family.link.inverse
        mu = inverse_link(eta)
        
        deviance = glm_results.deviance
        
        # delta g' is derivative of inverse link function
        derivative_inverse_link = glm_results.family.link.inverse_deriv
        dmu_deta = derivative_inverse_link(eta)
        
        rss = sum(glm_results.resid_response**2)
        llf = glm_results.llf
        
        z = eta + (y - mu)/dmu_deta
        W = np.diag((dmu_deta**2)/max(np.var(mu), 1e-8))
        
        # Tikhonov regularization
        
        r1 = X.T @ W @ X
        #r1 = r1 + self.tikhonov_lambda * np.eye(r1.shape[0])
        r2 = X.T @ W @ z
    
        return r1, r2, deviance, llf, rss, len(y)
    
    
    
class LikelihoodRatioTest:
    def __init__(self, t0: TestingRound, t1: TestingRound) -> None:
        
        assert t0.y_label == t1.y_label
        assert len(t0.X_labels) + 1 == len(t1.X_labels)
        self.t0 = t0
        self.t1 = t1
        
        self.y_label = self.t0.y_label
        
        self.s_labels = self.t0.X_labels
        self.x_label = list(set(self.t1.X_labels) - set(self.t0.X_labels))[0]
            
        #print(f"{t0}\n{t1}\n{'-'*20}\n{self.y_label, self.x_label, self.s_labels}")
        
        self.p_val = round(self._run_likelihood_test(),4)
        #self.p_val = round(self._run_f_test(),4)
        
    def __repr__(self):
        return f"LikelihoodRatioTest - y: {self.y_label}, x: {self.x_label}, S: {self.s_labels}, p: {self.p_val}"
    
    def _run_likelihood_test(self):
        
        # t1 should always encompass more regressors -> less client can fulfill this
        #assert len(self.t1.providing_clients) < len(self.t0.providing_clients)
        
        t0_fit_stats = self.t0.get_fit_stats(self.t1.providing_clients)
        t1_fit_stats = self.t1.get_fit_stats(self.t1.providing_clients)
        
        #assert t0_fit_stats['nobs'] == t1_fit_stats['nobs']
        
        t = -2*(t0_fit_stats['llf'] - t1_fit_stats['llf'])
        
        par0 = len(self.t0.X_labels) + 1 # + intercept
        par1 = len(self.t1.X_labels) + 1 # + intercept
        
        p_val = scipy.stats.chi2.sf(t, par1-par0)
        
        return p_val
        
    def _run_f_test(self):
        # t1 should always encompass more regressors -> less client can fulfill this
        #assert len(self.t1.providing_clients) < len(self.t0.providing_clients)
        
        t0_fit_stats = self.t0.get_fit_stats(self.t1.providing_clients)
        t1_fit_stats = self.t1.get_fit_stats(self.t1.providing_clients)
        
        rss0 = t0_fit_stats['rss']
        rss1 = t1_fit_stats['rss']
        par0 = len(self.t0.X_labels) + 1 # X + intercept
        par1 = len(self.t1.X_labels) + 1 # X + intercept
        nobs = t0_fit_stats['nobs']
        delta_rss = rss0 - rss1
        dfn = par1 - par0
        dfd = nobs - par1
        
        f = delta_rss*dfd/rss1/dfn
        
        p_val = scipy.stats.f.sf(f, dfn, dfd)
        
        return p_val
    
class SymmetricLikelihoodRatioTest:
    
    def __init__(self, lrt0: LikelihoodRatioTest, lrt1: LikelihoodRatioTest):
        
        assert lrt0.y_label == lrt1.x_label or lrt1.x_label is None
        assert lrt1.y_label == lrt0.x_label or lrt0.x_label is None
        
        
        #print(lrt0.s_labels, lrt1.s_labels)
        assert lrt0.s_labels.sort() == lrt1.s_labels.sort()
        
        self.label1 = lrt0.y_label
        self.label2 = lrt1.y_label
        self.conditioning_set = lrt0.s_labels
        
        self.lrt0: LikelihoodRatioTest = lrt0
        self.lrt1: LikelihoodRatioTest = lrt1
        
        self.p_val = min(2*min(self.lrt0.p_val, self.lrt1.p_val), max(self.lrt0.p_val, self.lrt1.p_val))
        
    def __repr__(self):
        #return f"SymmetricLikelihoodRatioTest - v0: {self.label1}, v1: {self.label2}, conditioning set: {self.conditioning_set}"
        return f"SymmetricLikelihoodRatioTest - v0: {self.label1}, v1: {self.label2}, conditioning set: {self.conditioning_set}, p: {self.p_val}\n\t-{self.lrt0}\n\t-{self.lrt1}"
    
    
# Helper Functions
    
def get_likelihood_tests(finished_rounds):
    tests = []
    
    for curr_round in finished_rounds:
        curr_y = curr_round.y_label
        curr_X = set(curr_round.X_labels)
        for x_var in curr_X:
            curr_conditioning_set = curr_X - {x_var}
            comparison_round = [t for t in finished_rounds if (set(t.X_labels) == curr_conditioning_set) and t.y_label == curr_y]
            if len(comparison_round) == 0:
                continue
            assert len(comparison_round) == 1
            comparison_round = comparison_round[0]
            tests.append(LikelihoodRatioTest(comparison_round, curr_round))
    
    return tests

def get_symmetric_likelihood_tests(finished_rounds):
    tests = []
    asymmetric_tests = get_likelihood_tests(finished_rounds)
    
    unique_tests = [t for t in asymmetric_tests if t.y_label < t.x_label]

    for test in unique_tests:
        swapped_test = [t for t in asymmetric_tests if (t.y_label == test.x_label) and (t.x_label == test.y_label) and (t.s_labels == test.s_labels)]
        if len(swapped_test) == 0:
            continue
        assert len(swapped_test) == 1
        swapped_test = swapped_test[0]
        tests.append(SymmetricLikelihoodRatioTest(test, swapped_test))
        
    return tests