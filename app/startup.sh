#!/bin/bash

if [ "$MODE" = "CLIENT" ]; then
    streamlit run streamlit/app.py --server.enableXsrfProtection false
elif [ "$MODE" = "SERVER" ]; then
    litestar --app-dir litestar run
elif [ "$MODE" = "HYBRID" ]; then
    streamlit run streamlit/app.py --server.enableXsrfProtection false & litestar --app-dir litestar run
else
    echo 'Please choose on of >CLIENT<, >SERVER<, or >HYBRID< as the MODE environment variable'
    exit 1
fi
