# rIOD

todo: algorithm descriptions not yet correct.

This project implements the IOD algorithm described in TODO.

A web app is provided which allows you to connect to a running server instance.  
This app runs algorithm 1 of the paper.
Once run, the results of multiple clients can be aggregated by the server, which runs algorithms 2 and 3 of the paper.

Additionally, the server code itself is provided, enabling you to host your own server for federated causal discovery.

Without code modifications, this application will only ever transmit p-values and PAG adjacency matrices over the network.  
The original dataset is kept completely private and is never transmitted over the network.  
(This can be easily checked inside the source code)


**Beware**: As of now, this client-server architecture communicates via http (_not_ https!).  
As such a malicious agent may spoof your identity or steal data that is transmitted over the network.

When hosting this application, you may use a reverse-proxy and your own SSL certificates to enable https.

## Setup

First, install docker and docker-compose.

Use the following commands to build and run the application:

* Run client and server on same machine:  
`docker-compose up`

* Run client or server only:  
`docker-compuse up client`  
`docker-compose up server`

* The client can be accessed via `localhost:8081` on the host machine.  
When hosting the client on a machine within the same network, use it's ip address and ensure proper connectivity between the two machines.

* The first step within the client is to connect to a server.  
A server hosted by us can be used with the following URL: `heiderlab.com:8080`

## Configuration

No major changes of this setup should be required.
All configurations can be changed in `docker-compose.yml`.

The only reasonable change is the port mapping, if your host machine requires specific ports to be exposed. 
