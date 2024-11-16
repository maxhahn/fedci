from typing import Dict
from data_structures import Connection, Room
import datetime

rooms: Dict[str, Room] = {}
connections: Dict[str, Connection] = {}

user2room: Dict[str, Room] = {}
user2connection: Dict[str, Connection] = {}

last_cleanse_time = datetime.datetime.now()
