
import numpy as np
import pickle
from typing import List
from itertools import chain, combinations

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.genmod.families import family

class TestingRound:
    iterations = 0
    last_deviance = None
    deviance = 0
    convergence_threshold = 1e-8
    
    def __init__(self, y_label, X_labels):
        self.y_label = y_label
        self.X_labels = X_labels
        self.beta = np.zeros(len(self.X_labels) + 1) # +1 for intercept
        
    def __eq__(self, t):
        assert type(t) == TestingRound
        return self.y_label == t.y_label and self.X_labels == t.X_labels   
    
    def __repr__(self):
        return f'TestingRound - y: {self.y_label}, X: {self.X_labels}, beta: {self.beta}, current iteration: {self.iterations}, current deviance: {abs(self.deviance)}, relative deviance change: {c if (c := self.get_relative_change_in_deviance()) is not None else "?"}' 
    
    def get_relative_change_in_deviance(self):
        if self.last_deviance is None:
            return None
        return abs(self.deviance - self.last_deviance) / (0.1 + abs(self.deviance))
    
    def aggregate_results(self, results):
        results1, results2, deviances = zip(*results)
        self.beta = np.linalg.inv(sum(results1)) @ sum(results2)
        self.last_deviance = self.deviance
        self.deviance = sum(deviances)
        self.iterations += 1
        
        return c < self.convergence_threshold if (c := self.get_relative_change_in_deviance()) is not None else False
    
class TestingEngine:
    testing_rounds = []
    finished_rounds = []
    
    def __init__(self, available_data, max_regressors=None, max_iterations=25, save_steps=10):
        self.available_data = available_data
        self.max_regressors = max_regressors
        self.max_iterations = max_iterations
        self.save_steps = save_steps
        
        _max_regressors = min(len(self.available_data), self.max_regressors+1) if self.max_regressors is not None else len(self.available_data)
        
        for e in available_data:
            set_of_regressors = available_data - {e}
            powerset_of_regressors = chain.from_iterable(combinations(set_of_regressors, r) for r in range(1,_max_regressors))
            self.testing_rounds.extend([TestingRound(y_label=e, X_labels=list(r)) for r in powerset_of_regressors])
            
        self.testing_rounds = sorted(self.testing_rounds, key=lambda key: len(key.X_labels))
        self.is_finished = len(self.testing_rounds) == 0
            
    def get_current_test_parameters(self):
        curr_testing_round = self.testing_rounds[0]
        return curr_testing_round.y_label, curr_testing_round.X_labels, curr_testing_round.beta
    
    def finish_current_test(self):
        self.finished_rounds.append(self.testing_rounds.pop(0))
        self.is_finished = len(self.testing_rounds) == 0
        
    def aggregate_results(self, results):
        has_converged = self.testing_rounds[0].aggregate_results(results)
        has_reached_max_iterations = self.testing_rounds[0].iterations >= self.max_iterations
        print(self.testing_rounds[0])
        if has_converged or has_reached_max_iterations:
            self.finish_current_test()
        
        
# TODO: current formula as Room State
# TODO: share current task among all clients -> those who can provide data do so, while the others sleep and retry


class Server:
    def __init__(self, clients):
        self.clients = clients
        self.available_data = set.union(*[set(c.data_labels) for c in self.clients.values()])
        self.testing_engine = TestingEngine(self.available_data, max_regressors=None)
        
    def run_tests(self):
        counter = 1
        while not self.testing_engine.is_finished:
            y_label, X_labels, beta = self.testing_engine.get_current_test_parameters()
            selected_clients = {id_: c for id_, c in self.clients.items() if set([y_label] + X_labels).issubset(c.data_labels)}
            # http response, to compute glm results for y regressed on X with beta
            results = [c.compute(y_label, X_labels, beta) for c in selected_clients.values()]
            self.testing_engine.aggregate_results(results)
            if counter % self.testing_engine.save_steps == 0:
                counter = 0
                with open('./testengine.ckp', 'wb') as f:
                    pickle.dump(self.testing_engine, f)
            counter += 1
                
    
class Client:
    def __init__(self, data):
        self.data = data
        self.data_labels = data.columns
        
        
    def compute(self, y_label: str, X_labels: List[str], beta):
        y = self.data[y_label]
        X = self.data[X_labels]
        
        X = X.to_numpy()
        X = sm.tools.add_constant(X)
                
        eta, mu, dmu_deta, deviance = self._init_compute(y,X,beta)

        z = eta + (y - mu)/dmu_deta
        W = np.diag((dmu_deta**2)/max(np.var(mu), 1e-8))
        
        r1 = X.T @ W @ X
        r2 = X.T @ W @ z
        
        return r1, r2, deviance
        
        
    def _init_compute(self, y, X, beta):
        glm_model = sm.GLM(y, X, family=family.Gaussian())
        normalized_cov_params = np.linalg.inv(X.T.dot(X))
        scale = glm_model.fit().scale
        glm_results = GLMResults(glm_model, beta, normalized_cov_params=normalized_cov_params, scale=None)
        
        # GLMResult with correct scale
        scale = glm_model.estimate_scale(glm_results.predict(which='linear'))
        glm_results = GLMResults(glm_model, beta, normalized_cov_params=normalized_cov_params, scale=scale)
        
        eta = glm_results.predict(which='linear')
        
        # g' is inverse of link function
        inverse_link = glm_results.family.link.inverse
        mu = inverse_link(eta)
        
        deviance = glm_results.deviance
        
        # delta g' is derivative of inverse link function
        derivative_inverse_link = glm_results.family.link.inverse_deriv
        dmu_deta = derivative_inverse_link(eta)
        return eta, mu, dmu_deta, deviance
    