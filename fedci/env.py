import os
DEBUG = 0 if (v:=os.getenv("DEBUG")) is None else int(v)
