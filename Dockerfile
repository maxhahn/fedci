# Use the official R base image from Docker Hub
FROM rocker/r-ver:4.4

# ,---.                 ,--.                       ,--------.             ,--.
# '   .-',--. ,--.,---.,-'  '-. ,---. ,--,--,--.    '--.  .--',---.  ,---. |  | ,---.
# `.  `-. \  '  /(  .-''-.  .-'| .-. :|        |       |  |  | .-. || .-. ||  |(  .-'
# .-'    | \   ' .-'  `) |  |  \   --.|  |  |  |       |  |  ' '-' '' '-' '|  |.-'  `)
# `-----'.-'  /  `----'  `--'   `----'`--`--`--'       `--'   `---'  `---' `--'`----'
#        `---'

RUN apt update && apt upgrade -y

RUN apt install -y cmake \
    libcurl4-openssl-dev \
    libv8-dev \
    libgsl-dev \
    libmagick++-dev

# ,------.     ,------.               ,--.
# |  .--. '    |  .--. ' ,--,--. ,---.|  |,-. ,--,--. ,---.  ,---.  ,---.
# |  '--'.'    |  '--' |' ,-.  || .--'|     /' ,-.  || .-. || .-. :(  .-'
# |  |\  \     |  | --' \ '-'  |\ `--.|  \  \\ '-'  |' '-' '\   --..-'  `)
# `--' '--'    `--'      `--`--' `---'`--'`--'`--`--'.`-  /  `----'`----'
#                                                     `---'

# Copy your R project files
COPY ./rIOD /rIOD
WORKDIR /rIOD

# install packages
RUN install2.r -e MXM pscl dagitty DOT rsvg BFF BiocManager
RUN R -e "BiocManager::install(c('graph', 'RBGL', 'Rgraphviz'))"
RUN install2.r -e rje pcalg jsonlite lavaan doFuture gtools

# install FCI Utils
RUN R CMD INSTALL imports/FCI.Utils_1.1.tar.gz

# build and install rIOD package
RUN R CMD build .
RUN R CMD INSTALL rIOD_1.0.tar.gz

# ,------.            ,--.  ,--.                        ,------.               ,--.
# |  .--. ',--. ,--.,-'  '-.|  ,---.  ,---. ,--,--,     |  .--. ' ,--,--. ,---.|  |,-. ,--,--. ,---.  ,---.  ,---.
# |  '--' | \  '  / '-.  .-'|  .-.  || .-. ||      \    |  '--' |' ,-.  || .--'|     /' ,-.  || .-. || .-. :(  .-'
# |  | --'   \   '    |  |  |  | |  |' '-' '|  ||  |    |  | --' \ '-'  |\ `--.|  \  \\ '-'  |' '-' '\   --..-'  `)
# `--'     .-'  /     `--'  `--' `--' `---' `--''--'    `--'      `--`--' `---'`--'`--'`--`--'.`-  /  `----'`----'
#          `---'                                                                              `---'

COPY ./app /app
WORKDIR /app

# get pip
RUN apt install -y pip
# Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install pandas polars graphviz rpy2 litestar[standard] streamlit extra-streamlit-components streamlit-extras streamlit-autorefresh
RUN pip install statsmodels scipy

# make startup script executable
RUN chmod +x startup.sh
# Draws config from env vars
CMD ./startup.sh
