import os

DEBUG = 0 if (v := os.getenv("DEBUG")) is None else int(v)
EXPAND_ORDINALS = 1 if (v := os.getenv("EXPAND_ORDINALS")) is None else int(v)
# TODO: fit_intercept=0 fails on no cond set, bc 0 features causes issue
# -> make exception handling for len 0 betas
FIT_INTERCEPT = 1 if (v := os.getenv("FIT_INTERCEPT")) is None else int(v)
CLIENT_HETEROGENIETY = 1 if (v := os.getenv("CLIENT_HETEROGENIETY")) is None else int(v)
LR = 1 if (v := os.getenv("LR")) is None else float(v)
RIDGE = 0 if (v := os.getenv("RIDGE")) is None else float(v)
OVR = 0 if (v := os.getenv("OVR")) is None else int(v)
