import pytest
import threading
import polars as pl
from fedci.client import ProxyClient, Client
from fedci.server import ProxyServer, Server

@pytest.fixture
def sample_data():
    data = {
        'A': [True, True, True, False, False, False, True, True, True, False],
        'B': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'C': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'D': [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0]
    }
    return pl.from_dict(data)

def test_local_server_single_client(sample_data):
    server = Server({'1': Client(sample_data)})
    results = server.run()
    for t in server.get_likelihood_ratio_tests():
        print(t)
    assert len(server.get_likelihood_ratio_tests()) == 32

# def test_local_server_multiple_clients(sample_data):
#     df1 = sample_data[:len(sample_data) // 2]
#     df2 = sample_data[len(sample_data) // 2:]

#     server = Server({'1': Client(df1), '2': Client(df2)})
#     results = server.run()
#     assert len(results) == 32

# def test_proxy_server_single_client(sample_data):
#     port = 18862
#     client = ProxyClient(sample_data)
#     server_thread = threading.Thread(target=client.start, args=(port,), daemon=True)
#     server_thread.start()

#     server_proxy = ProxyServer.builder() \
#         .add_client('localhost', port) \
#         .build()
#     results = server_proxy.server.run()

#     client.close()
#     assert len(results) == 32

# def test_proxy_server_multiple_clients(sample_data):
#     df1 = sample_data[:len(sample_data) // 2]
#     df2 = sample_data[len(sample_data) // 2:]

#     port1, port2 = 18862, 18863

#     client1 = ProxyClient(df1)
#     client2 = ProxyClient(df2)

#     t1 = threading.Thread(target=client1.start, args=(port1,), daemon=True)
#     t2 = threading.Thread(target=client2.start, args=(port2,), daemon=True)

#     t1.start()
#     t2.start()

#     server_proxy = ProxyServer.builder() \
#         .add_client('localhost', port1) \
#         .add_client('localhost', port2) \
#         .build()
#     results = server_proxy.server.run()

#     client1.close()
#     client2.close()

#     assert len(results) == 32

# def test_proxy_server_multiple_clients_0_regressors(sample_data):
#     df1 = sample_data[:len(sample_data) // 2]
#     df2 = sample_data[len(sample_data) // 2:]

#     port1, port2 = 18862, 18863

#     client1 = ProxyClient(df1)
#     client2 = ProxyClient(df2)

#     t1 = threading.Thread(target=client1.start, args=(port1,), daemon=True)
#     t2 = threading.Thread(target=client2.start, args=(port2,), daemon=True)

#     t1.start()
#     t2.start()

#     server_proxy = ProxyServer.builder() \
#         .set_max_regressors(0) \
#         .add_client('localhost', port1) \
#         .add_client('localhost', port2) \
#         .build()
#     results = server_proxy.server.run()

#     client1.close()
#     client2.close()

#     assert len(results) == 4

# def test_proxy_server_multiple_clients_1_regressor(sample_data):
#     df1 = sample_data[:len(sample_data) // 2]
#     df2 = sample_data[len(sample_data) // 2:]

#     port1, port2 = 18862, 18863

#     client1 = ProxyClient(df1)
#     client2 = ProxyClient(df2)

#     t1 = threading.Thread(target=client1.start, args=(port1,), daemon=True)
#     t2 = threading.Thread(target=client2.start, args=(port2,), daemon=True)

#     t1.start()
#     t2.start()

#     server_proxy = ProxyServer.builder() \
#         .set_max_regressors(1) \
#         .add_client('localhost', port1) \
#         .add_client('localhost', port2) \
#         .build()
#     results = server_proxy.server.run()

#     client1.close()
#     client2.close()

#     assert len(results) == 16

# def test_proxy_server_failing_connection(sample_data):
#     port1 = 18862

#     with pytest.raises(ConnectionRefusedError):
#         server_proxy = ProxyServer.builder() \
#             .add_client('localhost', port1) \
#             .build()
