from typing import Dict
import datetime
from typing import Optional
rooms: Dict[str, object] = {}
connections: Dict[str, object] = {}

user2room: Dict[str, object] = {}
user2connection: Dict[str, object] = {}

last_cleanse_time = datetime.datetime.now()
