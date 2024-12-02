

from .client import Client
from .testing import TestEngine
from .env import DEBUG
from typing import List, Dict

class Server():
    def __init__(self, clients: Dict[str, Client], max_regressors=None, test_targets=None):
        self.clients = clients
        self.client_schemas = {}
        self.schema = {}

        for client_id, client in self.clients.items():
            client_schema = client.get_data_schema()
            self.client_schemas[client_id] = client_schema
            for column, dtype in client_schema.items():
                if column not in self.schema:
                    self.schema[column] = dtype
                    continue
                assert self.schema[column] == dtype, f'Schema mismatch between clients detected for variable {column}!'

        self.category_expressions = {}
        self.ordinal_expressions = {}
        for client in self.clients.values():
            for feature, expressions in client.get_categorical_expressions().items():
                self.category_expressions[feature] = sorted(list(set(self.category_expressions.get(feature, [])).union(set(expressions))))
            for feature, levels in client.get_ordinal_expressions().items():
                self.ordinal_expressions[feature] = sorted(list(set(self.ordinal_expressions.get(feature, [])).union(set(levels))), key=lambda x: int(x.split('__ord__')[-1]))

        for client in self.clients.values(): client.provide_expressions(self.category_expressions, self.ordinal_expressions)

        self.test_engine: TestEngine = TestEngine(
            schema=self.schema,
            category_expressions=self.category_expressions,
            ordinal_expressions=self.ordinal_expressions,
            max_regressors=max_regressors,
            max_iterations=25,
            test_targets=test_targets
        )

    def run(self):
        while not self.test_engine.is_finished():
            required_labels = self.test_engine.get_currently_required_labels()

            selected_clients = {client_id: client for client_id, client in self.clients.items() if set(required_labels).issubset(self.client_schemas[client_id].keys())}
            assert len(selected_clients) > 0, f'No client is able to provide the data for {required_labels}'

            y_label, X_labels, beta = self.test_engine.get_current_test_parameters()
            results = {client_id: client.compute(y_label, X_labels, beta) for client_id, client in selected_clients.items()}
            self.test_engine.update_current_test(results)
        if DEBUG >= 1:
            print("*** All tests")
            for test in self.test_engine.tests:
                print(test)

    def get_tests(self):
        return self.test_engine.tests
