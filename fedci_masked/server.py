from typing import List

import rpyc

from .client import Client
from .env import ADDITIVE_MASKING, DEBUG
from .testing import TestEngine


class Server:
    def __init__(
        self,
        clients: List[Client],
        max_regressors=None,
        convergence_threshold=1e-3,
        max_iterations=25,
        _network_fetch_function=lambda x: x,
    ):
        self._network_fetch_function = _network_fetch_function
        self.clients = {client.get_id(): client for client in clients}
        assert len(self.clients) == len(clients), "Some Client IDs were duplicated"
        self.client_schemas = {}
        self.schema = {}

        for client_id, client in self.clients.items():
            client_schema = client.get_schema()
            self.client_schemas[client_id] = client_schema
            for column, dtype in client_schema.items():
                if column not in self.schema:
                    self.schema[column] = dtype
                    continue
                assert self.schema[column] == dtype, (
                    f"Schema mismatch between clients detected for variable {column}!"
                )

        self.category_expressions = {}
        self.ordinal_expressions = {}
        for client in self.clients.values():
            for feature, expressions in client.get_categorical_expressions().items():
                self.category_expressions[feature] = sorted(
                    list(
                        set(self.category_expressions.get(feature, [])).union(
                            set(expressions)
                        )
                    )
                )
            for feature, levels in client.get_ordinal_expressions().items():
                self.ordinal_expressions[feature] = sorted(
                    list(
                        set(self.ordinal_expressions.get(feature, [])).union(
                            set(levels)
                        )
                    ),
                    key=lambda x: int(x.split("__ord__")[-1]),
                )

        for client in self.clients.values():
            client.set_global_expressions(
                self.category_expressions, self.ordinal_expressions
            )
        for client in self.clients.values():
            client.set_clients(self.clients.copy())

        self.test_engine: TestEngine = TestEngine(
            schema=self.schema,
            category_expressions=self.category_expressions,
            ordinal_expressions=self.ordinal_expressions,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
        )

    def test(self, x: str, y: str, s: List[str]):
        if x > y:
            x, y = y, x
        self.test_engine.start_test(x, y, s)

        while not self.test_engine.is_finished():
            betas = self.test_engine.get_test_parameters()
            clients = self.clients.values()
            if ADDITIVE_MASKING:
                for client in clients:
                    client.exchange_masks(betas)
                for client in clients:
                    client.combine_masks(betas)
            updates = []
            for client in clients:
                update = client.compute(betas)
                updates.append(update)

            self.test_engine.update_parameters(updates)
        if DEBUG > 0:
            for key, beta in self.test_engine.test.get_betas().items():
                print(key)
                print(beta.tolist())
        return self.test_engine.get_result()

    def run(self, max_cond_size=None):
        finished_tests = []
        from itertools import chain, combinations

        def get_all_possible_tests(variables, max_cond_size):
            possible_tests = []

            if max_cond_size is None:
                max_cond_size = len(variables) - 2

            for y_var in variables:
                variables_without_y = variables - {y_var}
                for x_var in variables_without_y:
                    if y_var > x_var:
                        continue
                    set_of_possible_regressors = variables_without_y - {x_var}
                    powerset_of_regressors = chain.from_iterable(
                        combinations(set_of_possible_regressors, r)
                        for r in range(0, max_cond_size + 1)
                    )
                    for cond_set in powerset_of_regressors:
                        cond_set = list(cond_set)
                        possible_tests.append((y_var, x_var, cond_set))
            return possible_tests

        all_possible_tests = get_all_possible_tests(self.schema.keys(), max_cond_size)
        while len(all_possible_tests) > 0:
            current_test = all_possible_tests[0]
            del all_possible_tests[0]

            test_result = self.test(*current_test)
            if test_result is None:
                continue

            finished_tests.append(test_result)
        if DEBUG >= 1:
            print("*** All tests")
            for test in finished_tests:
                print(test)
        return finished_tests


class ProxyServerBuilder:
    def __init__(self, cls):
        self.clients = []
        self.cls = cls
        self.max_iterations = 25

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations
        return self

    def add_client(self, hostname, port):
        if (hostname, port) in self.clients:
            print("Client exists already")
            return self
        client = rpyc.connect(
            hostname, port, config={"allow_public_attrs": True, "allow_pickle": True}
        )
        self.clients.append(client)
        return self

    def build(self):
        return self.cls(
            self.clients,
            max_iterations=self.max_iterations,
        )


class ProxyServer:
    @classmethod
    def builder(cls, **kwargs):
        return ProxyServerBuilder(cls, **kwargs)

    def __init__(self, clients, max_iterations):
        self.clients = {i: c.root for i, c in enumerate(clients)}
        self.server = Server(
            self.clients,
            _network_fetch_function=rpyc.classic.obtain,
            max_iterations=max_iterations,
        )

    def __getattr__(self, name):
        return getattr(self.server, name)

    def test(self, x, y, s):
        return self.server.test(x, y, s)

    def run(self, max_cond_size=None):
        return self.server.run(max_cond_size)
