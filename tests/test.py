import threading

import polars as pl
import pytest
import scipy

from fedci.client import Client, ProxyClient
from fedci.server import ProxyServer, Server


@pytest.fixture
def sample_data():
    data = {
        "A": [
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
        ],
        "B": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "C": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "B",
            "B",
        ],
        "D": [
            1.0,
            1.0,
            2.0,
            2.0,
            3.0,
            3.0,
            1.0,
            1.0,
            2.0,
            2.0,
            3.0,
            3.0,
            3.5,
            1.5,
            2.5,
        ],
    }
    return pl.from_dict(data)


def test_local_server_single_client_single_test(sample_data):
    server = Server({"1": Client(sample_data)})

    result = server.test("A", "B", ["C"])

    assert result.v0 == "A"
    assert result.v1 == "B"
    assert result.conditioning_set == ["C"]


def test_local_server_single_client(sample_data):
    server = Server({"1": Client(sample_data)})

    results = server.run()

    # expected tests is (n over 2) x (n-2)^2, pick two variables X and Y and all others can either be in conditioning set or not
    num_cols = len(sample_data.columns)
    assert len(results) == scipy.special.binom(num_cols, 2) * (num_cols - 2) ** 2


# this test covers diverging parameters
def test_local_server_single_client_partial_data(sample_data):
    subdata = sample_data.head(10)
    server = Server({"1": Client(subdata)})

    results = server.run()

    # expected tests is (n over 2) x (n-2)^2, pick two variables X and Y and all others can either be in conditioning set or not
    num_cols = len(sample_data.columns)
    assert len(results) == scipy.special.binom(num_cols, 2) * (num_cols - 2) ** 2


# this test covers no intercept no conditioning set tests
def test_local_server_single_client_single_test_no_intercept(sample_data):
    server = Server({"1": Client(sample_data)})

    result = server.test("C", "D", [])

    assert result.v0 == "B"
    assert result.v1 == "C"
    assert result.conditioning_set == []


def test_local_server_multiple_clients(sample_data):
    df1 = sample_data[: len(sample_data) // 2]
    df2 = sample_data[len(sample_data) // 2 :]

    server = Server({"1": Client(df1), "2": Client(df2)})
    results = server.run()

    num_cols = len(sample_data.columns)
    assert len(results) == scipy.special.binom(num_cols, 2) * (num_cols - 2) ** 2


def test_local_server_multiple_clients_cond_size_0(sample_data):
    df1 = sample_data[: len(sample_data) // 2]
    df2 = sample_data[len(sample_data) // 2 :]

    server = Server({"1": Client(df1), "2": Client(df2)})
    results = server.run(max_cond_size=0)

    for r in sorted(results):
        print(r)

    num_cols = len(sample_data.columns)
    assert len(results) == (
        scipy.special.binom(num_cols, 2)  # all tests for cond size 0
    )


def test_local_server_multiple_clients_partial_overlap(sample_data):
    df1 = sample_data[: len(sample_data) // 2].select("A", "B", "C")
    df2 = sample_data[len(sample_data) // 2 :].select("B", "C", "D")

    server = Server({"1": Client(df1), "2": Client(df2)})
    results = server.run()

    c1_num_cols = len(df1.columns)
    c2_num_cols = len(df2.columns)

    # one possible test on shared variables
    # n over 2 for pairs of variables choices with either a variable in cond set or not (x2), for both clients,
    #   subtracting the test that can be performed on the shared vars
    assert (
        len(results)
        == 1
        + (scipy.special.binom(c1_num_cols, 2) * 2)
        - 1
        + (scipy.special.binom(c2_num_cols, 2) * 2)
        - 1
    )


def test_local_server_multiple_clients_cond_size_1(sample_data):
    df1 = sample_data[: len(sample_data) // 2]
    df2 = sample_data[len(sample_data) // 2 :]

    server = Server({"1": Client(df1), "2": Client(df2)})
    results = server.run(max_cond_size=1)

    num_cols = len(sample_data.columns)
    assert len(results) == (
        scipy.special.binom(num_cols, 2) * (num_cols - 2)
    ) + scipy.special.binom(num_cols, 2)  # all tests for 0 and all tests for 1


def test_proxy_server_multiple_clients_single_test(sample_data):
    df1 = sample_data[: len(sample_data) // 2]
    df2 = sample_data[len(sample_data) // 2 :]

    port1, port2 = 18862, 18863

    client1 = ProxyClient(df1)
    client2 = ProxyClient(df2)

    t1 = threading.Thread(target=client1.start, args=(port1,), daemon=True)
    t2 = threading.Thread(target=client2.start, args=(port2,), daemon=True)

    t1.start()
    t2.start()

    server_proxy = (
        ProxyServer.builder()
        .add_client("localhost", port1)
        .add_client("localhost", port2)
        .build()
    )
    result = server_proxy.test("A", "B", ["C"])

    client1.close()
    client2.close()

    assert result.v0 == "A"
    assert result.v1 == "B"
    assert result.conditioning_set == ["C"]


def test_proxy_server_multiple_clients(sample_data):
    df1 = sample_data[: len(sample_data) // 2]
    df2 = sample_data[len(sample_data) // 2 :]

    port1, port2 = 18862, 18863

    client1 = ProxyClient(df1)
    client2 = ProxyClient(df2)

    t1 = threading.Thread(target=client1.start, args=(port1,), daemon=True)
    t2 = threading.Thread(target=client2.start, args=(port2,), daemon=True)

    t1.start()
    t2.start()

    server_proxy = (
        ProxyServer.builder()
        .add_client("localhost", port1)
        .add_client("localhost", port2)
        .build()
    )
    results = server_proxy.run()

    client1.close()
    client2.close()

    num_cols = len(sample_data.columns)
    assert len(results) == scipy.special.binom(num_cols, 2) * (num_cols - 2) ** 2


def test_proxy_server_multiple_clients_1_regressor(sample_data):
    df1 = sample_data[: len(sample_data) // 2]
    df2 = sample_data[len(sample_data) // 2 :]

    port1, port2 = 18862, 18863

    client1 = ProxyClient(df1)
    client2 = ProxyClient(df2)

    t1 = threading.Thread(target=client1.start, args=(port1,), daemon=True)
    t2 = threading.Thread(target=client2.start, args=(port2,), daemon=True)

    t1.start()
    t2.start()

    server_proxy = (
        ProxyServer.builder()
        .add_client("localhost", port1)
        .add_client("localhost", port2)
        .build()
    )
    results = server_proxy.run(1)

    client1.close()
    client2.close()

    num_cols = len(sample_data.columns)
    assert len(results) == (
        scipy.special.binom(num_cols, 2) * (num_cols - 2)
    ) + scipy.special.binom(num_cols, 2)  # all tests for 0 and all tests for 1
