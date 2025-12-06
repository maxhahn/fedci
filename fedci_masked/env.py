import os

DEBUG = 0 if (v := os.getenv("DEBUG")) is None else int(v)
FIT_INTERCEPT = 1 if (v := os.getenv("FIT_INTERCEPT")) is None else int(v)
CLIENT_HETEROGENIETY = 1 if (v := os.getenv("CLIENT_HETEROGENIETY")) is None else int(v)
ADDITIVE_MASKING = 1 if (v := os.getenv("ADDITIVE_MASKING")) is None else int(v)
