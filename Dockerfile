# Use the official R base image from Docker Hub
FROM rocker/r-ver:4.4

RUN apt update && apt upgrade -y

RUN apt install -y cmake \
    libcurl4-openssl-dev \
    libv8-dev \
    libgsl-dev \
    libmagick++-dev

# Copy your R project files
COPY ./rIOD /rIOD
WORKDIR /rIOD

# install packages
RUN install2.r -e MXM pscl dagitty DOT rsvg BFF BiocManager
RUN R -e "BiocManager::install(c('graph', 'RBGL', 'Rgraphviz'))"
RUN install2.r -e rje pcalg jsonlite lavaan doFuture gtools


# install FCI Utils
RUN R CMD INSTALL imports/FCI.Utils_1.0.tar.gz 

# build and install rIOD package
RUN R CMD build . 
RUN R CMD INSTALL rIOD_1.0.tar.gz 

ENTRYPOINT /bin/bash
