from dataclasses import dataclass

from typing import Annotated, List, Dict, Set

from litestar import Litestar, MediaType, Response, post, get
from litestar.exceptions import HTTPException
from litestar.params import Body
import datetime

import uuid


@dataclass
class Connection:
    id: str
    username: str
    last_request_time: datetime.datetime

@dataclass
class Room:
    id: str
    name: str
    owner_name: str
    is_locked: bool
    users: Set[str]
    
@dataclass
class UserDTO:
    id: str
    username: str
    def __init__(self, conn: Connection):
        self.id = conn.id
        self.username = conn.username
        
@dataclass
class RoomDTO:
    id: str
    name: str
    owner_name: str
    is_locked: bool
    def __init__(self, room: Room):
        self.id = room.id
        self.name = room.name
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        
@dataclass
class CheckInRequest:
    username: str
        
@dataclass
class BasicRequest:
    id: str
    username: str
    
@dataclass
class RoomCreationRequest(BasicRequest):
    room_name: str

rooms: Dict[str, Room] = []
connections: Dict[str, Connection] = {}
user2room:  Dict[str, Room] = {}

def validate_user_request(id: str, username: str):
    if id not in connections:
        return False
    
    if connections[id].username != username:
        return False
    
    connections[id].last_request_time = datetime.datetime.now()
    return True
    

@get("/health-check")
async def health_check() -> Response:
    return Response(
        media_type=MediaType.TEXT,
        content='Hello there!',
        status_code=200
    )

@post("/check-in")
async def check_in(request: Annotated[CheckInRequest, Body(title='Check-In', description='Check in to server with a username of choice')]) -> Response:
    # guarantee unused new id
    new_id = uuid.uuid4()
    while new_id in connections:
        new_id = uuid.uuid4()
        
    username = request.username
        
    username_offset = 1
    occupied_usernames = [c.username for c in connections]
    new_username = username
    while new_username in occupied_usernames:
        new_username = username + f'({username_offset})'
        username_offset += 1
        
    conn = Connection(new_id, new_username, datetime.datetime.now())
    connections[new_id] = conn

    return Response(
        media_type=MediaType.JSON,
        content=UserDTO(conn),
        status_code=200
        )

# list rooms
@get("/rooms")
async def get_rooms(request: Annotated[BasicRequest, Body(title='List Rooms', description='Get a list of all existing rooms')]) -> Response:
    if not validate_user_request(request.id, request.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    return Response(
        media_type=MediaType.JSON,
        content=[RoomDTO(room) for room in rooms],
        status_code=200
        )

# join room
@get("/rooms/join/{room_id:str}")
async def get_room(request: BasicRequest, room_id: str) -> Response:
    if not validate_user_request(request.id, request.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if request.username in user2room:
        raise HTTPException(detail='You are already assigned a room', status_code=403)
    if room_id not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_id]
    room.users.add(request.username)
    user2room[request.username] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDTO(room),
        status_code=200
        )
    
# leave room
@get("/rooms/leave/{room_id:str}")
async def leave_room(request: BasicRequest, room_id: str) -> Response:
    if not validate_user_request(request.id, request.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_id not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_id]
    
    if request.username not in room.users:
        raise HTTPException(detail='You are not in the room', status_code=403)
    
    room.users.remove(request.username)
    del user2room[request.username]
    
    return Response(
        media_type=MediaType.TEXT,
        content="You left the room",
        status_code=200
        )
    
# get room info
@get("/rooms/{room_id:str}")
async def leave_room(request: BasicRequest, room_id: str, username_to_kick: str) -> Response:
    if not validate_user_request(request.id, request.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_id not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_id]

    return Response(
        media_type=MediaType.JSON,
        content=RoomDTO(room),
        status_code=200
        )
    
# kick user from room
@get("/rooms/{room_id:str}/kick/{username_to_kick:str}")
async def leave_room(request: BasicRequest, room_id: str, username_to_kick: str) -> Response:
    if not validate_user_request(request.id, request.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_id not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_id]
    
    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != request.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)
    
    room.users.remove(username_to_kick)
    del user2room[username_to_kick]
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDTO(room),
        status_code=200
        )

# create room
@post("/rooms")
async def create_room(request: RoomCreationRequest) -> Response:
    if not validate_user_request(request.id, request.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    
    room_id = uuid.uuid4()
    while room_id in rooms:
        room_id = uuid.uuid4()
    room_owner = request.username
    room_name = request.room_name
    
    room_name_offset = 1
    new_room_name = room_name
    occupied_room_names = [r.name for r in rooms]
    while new_room_name in occupied_room_names:
        new_room_name = room_name + f'({room_name_offset})'
        room_name_offset += 1
    
    room = Room(id=room_id,
                name=new_room_name,
                owner_name=room_owner,
                is_locked=True,
                users=[connections[room_owner]]
                )
    
    rooms[room_id] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDTO(room),
        status_code=200
        )


app = Litestar([health_check])