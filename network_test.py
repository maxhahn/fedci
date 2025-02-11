from fedci.client import ProxyClient, Client
from fedci.server import ProxyServer, Server

import polars as pl
df = pl.read_parquet('test.parquet')

x = 2
if x == 1:
    s = Server({'1': Client(df)})
    r = s.run()
    print(r)
elif x == 2:
    import threading

    port = 18862  # Each client should have a different port in a real setup
    print(f"Client started on port {port}...")
    client = ProxyClient(df)
    server_thread = threading.Thread(target=client.start, args=(port,), daemon=True)
    server_thread.start()

    server_proxy = ProxyServer.builder().set_max_regressors(0).add_client('localhost', port).build()
    results = server_proxy.server.run()
    client.close()
    print(results)
elif x == 3:
    df1 = df[:len(df)//2]
    df2 = df[len(df)//2:]

    import threading

    port1 = 18862  # Each client should have a different port in a real setup
    print(f"Client started on port {port1}...")
    client1 = ProxyClient(df1)
    t1 = threading.Thread(target=client1.start, args=(port1,), daemon=True)
    t1.start()

    port2 = 18863  # Each client should have a different port in a real setup
    print(f"Client started on port {port2}...")
    client2 = ProxyClient(df2)
    t2 = threading.Thread(target=client2.start, args=(port2,), daemon=True)
    t2.start()

    server_proxy = ProxyServer.builder().add_client('localhost', port1).add_client('localhost', port2).build()
    results = server_proxy.server.run()
    print(results)

    client1.close()
    client2.close()
