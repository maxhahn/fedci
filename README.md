# About this Project

This projects aims to create an accessible platform and user interface to obtain causal knowledge from distributed datasets.  
Different parties can share key insights about their data without compromising their privacy.  
Results come in the form of causal graphs (PAGs).

There are two main algorithm this paper supports:  
* rIOD
* FedGLM

## General

This application is made up of two main components.  
There is a streamlit UI and a litestar server.

Via the streamlit UI, one can connect to an existing server instance and run distributed/federated algorithms with peers connected to the same server.

It is designed to be easily self-hostable, and is fully contained within a docker container.

**Beware**: As of now, this client-server architecture communicates via http (_not_ https!).  
As such a malicious agent may spoof your identity or steal data that is transmitted over the network.

When hosting this application, you may use a reverse-proxy and your own SSL certificates to enable https.

## rIOD

This project implements the IOD algorithm created by [Tillman and Spirtes](http://proceedings.mlr.press/v15/tillman11a.html).

When running rIOD, (conditional) independence tests are performed and the resulting p-values are transmitted to the server.  
On the server-side, these p-values are aggregated to give insights about the independences in the distributed dataset.

rIOD only supports numerical (float) features.
As per the IOD algorithm, not all participating parties have to have identical features in their dataset.  

Without code modifications, this application will only ever transmit p-values and PAG adjacency matrices over the network.  
The original dataset is kept completely private and is never transmitted over the network.  
(This can be easily checked inside the source code)

## FedGLM

This project implemens an algorithm (titled FedGLM for now) which utilizes federated learning to create linear models, which are then used for likelihood-ratio tests in order to obtain independence information about the dataset.

FedGLM supports numerical (float), categorical (string), and ordinal (int) features.
Similar to IOD, here as well, not all participating parties have to have the exact same feature set.

This algorithm requires the transmission of the expression levels of ordinal and categorical variables.
Additionaly, when the algorithm is running, linear model coefficients are transmitted, as well as matrices that do not contain recreatable information about the data (see algorithm 2 in [Cellamare et al.](https://www.mdpi.com/1999-4893/15/7/243))
As such, data privacy is preserved.

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
