import os
DEBUG = 0 if (v:=os.getenv("DEBUG")) is None else int(v)
LOG_R = 0 if (v:=os.getenv("LOG_R")) is None else int(v)
EXPAND_ORDINALS = 0 if (v:=os.getenv("EXPAND_ORDINALS")) is None else int(v)
