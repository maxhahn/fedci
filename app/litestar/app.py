from dataclasses import dataclass

from typing import Annotated, List, Dict, Set, Optional

from litestar import Litestar, MediaType, Response, post, get
from litestar.exceptions import HTTPException
from litestar.params import Body

import datetime
import uuid

from collections import OrderedDict
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import pickle
import base64


# ,------.          ,--.  ,--.  ,--.  ,--.               
# |  .---',--,--, ,-'  '-.`--',-'  '-.`--' ,---.  ,---.  
# |  `--, |      \'-.  .-',--.'-.  .-',--.| .-. :(  .-'  
# |  `---.|  ||  |  |  |  |  |  |  |  |  |\   --..-'  `) 
# `------'`--''--'  `--'  `--'  `--'  `--' `----'`----'  

@dataclass
class Connection:
    id: str
    username: str
    last_request_time: datetime.datetime
    data: Optional[pd.DataFrame]=None
    data_labels: Optional[List[str]]=None

@dataclass
class Room:
    name: str
    owner_name: str
    password: str
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    users: Set[str]
    user_provided_labels: Dict[str, List[str]]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    user_results: Dict[str, List[List[int]]]
    user_labels: Dict[str, List[str]]
    
# ,------. ,--------. ,-----.         
# |  .-.  \'--.  .--''  .-.  ' ,---.  
# |  |  \  :  |  |   |  | |  |(  .-'  
# |  '--'  /  |  |   '  '-'  '.-'  `) 
# `-------'   `--'    `-----' `----'  
    
@dataclass
class UserDTO:
    id: str
    username: str
    submitted_data: bool
    data_labels: Optional[List[str]]
    def __init__(self, conn: Connection):
        self.id = conn.id
        self.username = conn.username
        self.submitted_data = conn.data is not None
        self.data_labels = conn.data_labels if self.submitted_data else None
        
@dataclass
class RoomDTO:
    name: str
    owner_name: str
    is_locked: bool
    def __init__(self, room: Room):
        self.name = room.name
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        
@dataclass
class RoomDetailsDTO:
    name: str
    owner_name: str
    is_locked: bool
    is_hidden: bool
    is_processing: bool
    is_finished: bool
    is_protected: bool
    users: List[str]
    user_provided_labels: Dict[str, List[str]]
    result: List[List[List[int]]]
    result_labels: List[List[str]]
    private_result: List[List[int]]
    private_labels: List[str]
    
    def __init__(self, room: Room, requesting_user: str=None):
        self.name = room.name
        self.owner_name = room.owner_name
        self.is_locked = room.is_locked
        self.is_hidden = room.is_hidden
        self.is_processing = room.is_processing
        self.is_finished = room.is_finished
        self.is_protected = room.password is not None
        self.users = sorted(list(room.users))
        self.user_provided_labels = room.user_provided_labels
        self.result = room.result
        self.result_labels = room.result_labels
        self.private_result = room.user_results[requesting_user] if room.user_results is not None and requesting_user in room.user_results else None
        self.private_labels = room.user_labels[requesting_user] if room.user_labels is not None and requesting_user in room.user_labels else None
        
# ,------.                                      ,--.          
# |  .--. ' ,---.  ,---. ,--.,--. ,---.  ,---.,-'  '-. ,---.  
# |  '--'.'| .-. :| .-. ||  ||  || .-. :(  .-''-.  .-'(  .-'  
# |  |\  \ \   --.' '-' |'  ''  '\   --..-'  `) |  |  .-'  `) 
# `--' '--' `----' `-|  | `----'  `----'`----'  `--'  `----'  
#                    `--' 

@dataclass
class CheckInRequest:
    username: str
        
@dataclass
class BasicRequest:
    id: str
    username: str
    
@dataclass
class ChangeUsernameRequest(BasicRequest):
    new_username: str
    
@dataclass
class RoomCreationRequest(BasicRequest):
    room_name: str
    password: str
    
@dataclass
class JoinRoomRequest(BasicRequest):
    password: str
    
@dataclass
class DataSubmissionRequest(BasicRequest):
    data: str
    data_labels: List[str]
    
@dataclass
class IODExecutionRequest(BasicRequest):
    alpha: float
    

# ,------.            ,--.               ,---.   ,--.                         ,--.                                
# |  .-.  \  ,--,--.,-'  '-. ,--,--.    '   .-',-'  '-.,--.--.,--.,--. ,---.,-'  '-.,--.,--.,--.--. ,---.  ,---.  
# |  |  \  :' ,-.  |'-.  .-'' ,-.  |    `.  `-.'-.  .-'|  .--'|  ||  || .--''-.  .-'|  ||  ||  .--'| .-. :(  .-'  
# |  '--'  /\ '-'  |  |  |  \ '-'  |    .-'    | |  |  |  |   '  ''  '\ `--.  |  |  '  ''  '|  |   \   --..-'  `) 
# `-------'  `--`--'  `--'   `--`--'    `-----'  `--'  `--'    `----'  `---'  `--'   `----' `--'    `----'`----' 

rooms: Dict[str, Room] = {}
connections: Dict[str, Connection] = {}
user2room: Dict[str, Room] = {}
user2connection: Dict[str, Connection] = {}
last_cleanse_time = datetime.datetime.now()

# ,--.  ,--.       ,--.                             
# |  '--'  | ,---. |  | ,---.  ,---. ,--.--. ,---.  
# |  .--.  || .-. :|  || .-. || .-. :|  .--'(  .-'  
# |  |  |  |\   --.|  || '-' '\   --.|  |   .-'  `) 
# `--'  `--' `----'`--'|  |-'  `----'`--'   `----'  
#                      `--'                         

def validate_user_request(id: str, username: str):
    if id not in connections:
        return False
    
    if connections[id].username != username:
        return False
    
    connections[id].last_request_time = datetime.datetime.now()
    return True
    
def cleanse_inactive_users(curr_time):
    for id, conn in connections.items():
        if (curr_time-conn.last_request_time).total_seconds() > 60*60*3:
            username = conn.username
            # if user is in room
            if username in user2room:
                # remove from room
                user2room[username].users.remove(username)
                # if room is empty -> remove room
                if len(user2room[username].users) == 0:
                    del rooms[user2room[username]]
                del user2room[username]
            # remove connection lookup and connection
            del user2connection[username]
            del conn[id]
            
# ,--.  ,--.               ,--.  ,--.  ,--.          ,-----.,--.                  ,--.     
# |  '--'  | ,---.  ,--,--.|  |,-'  '-.|  ,---.     '  .--./|  ,---.  ,---.  ,---.|  |,-.  
# |  .--.  || .-. :' ,-.  ||  |'-.  .-'|  .-.  |    |  |    |  .-.  || .-. :| .--'|     /  
# |  |  |  |\   --.\ '-'  ||  |  |  |  |  | |  |    '  '--'\|  | |  |\   --.\ `--.|  \  \  
# `--'  `--' `----' `--`--'`--'  `--'  `--' `--'     `-----'`--' `--' `----' `---'`--'`--' 

@get("/health-check")
async def health_check() -> Response:
    global last_cleanse_time
    curr_time = datetime.datetime.now()
    if (curr_time - last_cleanse_time).total_seconds() > 60*20:
        cleanse_inactive_users(curr_time)
        last_cleanse_time = curr_time
    
    return Response(
        media_type=MediaType.TEXT,
        content='Hello there!',
        status_code=200
    )
    
#                                                                                                         ,--.                                                 
#  ,-----.,--.,--.                 ,--.       ,-----.,--.                  ,--.          ,--.            /  /,------.                                          
# '  .--./|  |`--' ,---. ,--,--, ,-'  '-.    '  .--./|  ,---.  ,---.  ,---.|  |,-.,-----.|  |,--,--,    /  / |  .--. ' ,---. ,--,--,  ,--,--.,--,--,--. ,---.  
# |  |    |  |,--.| .-. :|      \'-.  .-'    |  |    |  .-.  || .-. :| .--'|     /'-----'|  ||      \  /  /  |  '--'.'| .-. :|      \' ,-.  ||        || .-. : 
# '  '--'\|  ||  |\   --.|  ||  |  |  |      '  '--'\|  | |  |\   --.\ `--.|  \  \       |  ||  ||  | /  /   |  |\  \ \   --.|  ||  |\ '-'  ||  |  |  |\   --. 
#  `-----'`--'`--' `----'`--''--'  `--'       `-----'`--' `--' `----' `---'`--'`--'      `--'`--''--'/  /    `--' '--' `----'`--''--' `--`--'`--`--`--' `----' 
#                                                                                                   `--'

@post("/check-in")
async def check_in(data: Annotated[CheckInRequest, Body(title='Check-In', description='Check in to server with a username of choice')]) -> Response:
    
    if data.username is None or len(data.username.replace(r'\s', '')) == 0:
        raise HTTPException(detail='Username is not accepted', status_code=400)
    
    # guarantee unused new id
    new_id = uuid.uuid4()
    while new_id in connections:
        new_id = uuid.uuid4()
    new_id = str(new_id)
        
    username = data.username
        
    username_offset = 1
    occupied_usernames = [c.username for c in connections.values()]
    new_username = username
    while new_username in occupied_usernames:
        new_username = username + f' ({username_offset})'
        username_offset += 1
        
    conn = Connection(new_id, new_username, datetime.datetime.now())
    connections[new_id] = conn
    user2connection[new_username] = conn

    return Response(
        media_type=MediaType.JSON,
        content=UserDTO(conn),
        status_code=200
        )
    
@post("/change-name")
async def change_name(data: Annotated[ChangeUsernameRequest, Body(title='Change Name', description='Change displayed username')]) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    
    if data.username is None or len(data.username.replace(r'\s', '')) == 0:
        raise HTTPException(detail='Username is not accepted', status_code=400)
        
    username_offset = 1
    occupied_usernames = [c.username for c in connections.values()]
    new_username = data.new_username
    while new_username in occupied_usernames:
        new_username = data.new_username + f' ({username_offset})'
        username_offset += 1
        
    connections[data.id].username = new_username
    conn = connections[data.id]

    return Response(
        media_type=MediaType.JSON,
        content=UserDTO(conn),
        status_code=200
        )
    
#                                                                          ,--.                                                                                     
#  ,----.            ,--.      ,------.                                   /  /,------.                             ,------.           ,--.          ,--.,--.        
# '  .-./    ,---. ,-'  '-.    |  .--. ' ,---.  ,---. ,--,--,--. ,---.   /  / |  .--. ' ,---.  ,---. ,--,--,--.    |  .-.  \  ,---. ,-'  '-. ,--,--.`--'|  | ,---.  
# |  | .---.| .-. :'-.  .-'    |  '--'.'| .-. || .-. ||        |(  .-'  /  /  |  '--'.'| .-. || .-. ||        |    |  |  \  :| .-. :'-.  .-'' ,-.  |,--.|  |(  .-'  
# '  '--'  |\   --.  |  |      |  |\  \ ' '-' '' '-' '|  |  |  |.-'  `)/  /   |  |\  \ ' '-' '' '-' '|  |  |  |    |  '--'  /\   --.  |  |  \ '-'  ||  ||  |.-'  `) 
#  `------'  `----'  `--'      `--' '--' `---'  `---' `--`--`--'`----'/  /    `--' '--' `---'  `---' `--`--`--'    `-------'  `----'  `--'   `--`--'`--'`--'`----'  
#                                                                    `--'  

# list rooms
@post("/rooms")
async def get_rooms(data: Annotated[BasicRequest, Body(title='List Rooms', description='Get a list of all existing rooms')]) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    return Response(
        media_type=MediaType.JSON,
        content=[RoomDTO(room) for room in rooms.values() if not room.is_hidden],
        status_code=200
        )
    
# get room info
@post("/rooms/{room_name:str}")
async def get_room(data: BasicRequest, room_name: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_name]
    
    if data.username not in room.users:
        raise HTTPException(detail='You are not in this room', status_code=403)

    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )
    
# ,------.                             ,--.          ,--.                                ,--.  ,--.                
# |  .--. ' ,---.  ,---. ,--,--,--.    |  |,--,--, ,-'  '-. ,---. ,--.--. ,--,--. ,---.,-'  '-.`--' ,---. ,--,--,  
# |  '--'.'| .-. || .-. ||        |    |  ||      \'-.  .-'| .-. :|  .--'' ,-.  || .--''-.  .-',--.| .-. ||      \ 
# |  |\  \ ' '-' '' '-' '|  |  |  |    |  ||  ||  |  |  |  \   --.|  |   \ '-'  |\ `--.  |  |  |  |' '-' '|  ||  | 
# `--' '--' `---'  `---' `--`--`--'    `--'`--''--'  `--'   `----'`--'    `--`--' `---'  `--'  `--' `---' `--''--' 
    
# create room
@post("/rooms/create")
async def create_room(data: RoomCreationRequest) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    
    room_owner = data.username
    room_name = data.room_name
    
    room_name_offset = 1
    new_room_name = room_name
    occupied_room_names = [r.name for r in rooms.values()]
    while new_room_name in occupied_room_names:
        new_room_name = room_name + f' ({room_name_offset})'
        room_name_offset += 1
    
    room = Room(name=new_room_name,
                owner_name=room_owner,
                password=data.password,
                is_locked=False,
                is_hidden=False,
                is_processing=False,
                is_finished=False,
                users={room_owner},
                user_provided_labels={room_owner: connections[data.id].data_labels},
                result=None,
                result_labels=None,
                user_results={},
                user_labels={}
                )
    
    rooms[new_room_name] = room
    user2room[room_owner] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# join room
@post("/rooms/{room_name:str}/join")
async def join_room(data: JoinRoomRequest, room_name: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if data.username in user2room:
        raise HTTPException(detail='You are already assigned a room', status_code=403)
    if connections[data.id].data is None:
        raise HTTPException(detail='You have to upload data before joining a room', status_code=403)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    
    room = rooms[room_name]
    
    if room.is_locked:
        raise HTTPException(detail='The room is locked', status_code=403)
    if room.password != data.password:
        raise HTTPException(detail='Incorrect password', status_code=403)
    
    room.users.add(data.username)
    room.user_provided_labels[data.username] = connections[data.id].data_labels
    user2room[data.username] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )
    
# leave room
@post("/rooms/{room_name:str}/leave")
async def leave_room(data: BasicRequest, room_name: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_name]
    
    if data.username not in room.users:
        raise HTTPException(detail='You are not in the room', status_code=403)
    
    room.users.remove(data.username)
    del user2room[data.username]
    # just remove room if it is empty
    if len(room.users) == 0:
        del rooms[room_name]
        return Response(
            media_type=MediaType.TEXT,
            content="You left the room",
            status_code=200
            )
    
    del room.user_provided_labels[data.username]
    if room.is_finished:
        del room.user_results[data.username]
        
    if data.username == room.owner_name:
        room.owner_name = room.users[0]
        
    rooms[room_name] = room # reassign
    
    return Response(
        media_type=MediaType.TEXT,
        content="You left the room",
        status_code=200
        )
    
# kick user from room
@post("/rooms/{room_name:str}/kick/{username_to_kick:str}")
async def kick_user_from_room(data: BasicRequest, room_name: str, username_to_kick: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_name]
    if room.is_finished:
        raise HTTPException(detail='Cannot kick users once IOD was executed', status_code=403)
    if room.is_processing:
        raise HTTPException(detail='Cannot kick while server processes the rooms data', status_code=403)
    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != data.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)
    if room.owner_name == username_to_kick:
        raise HTTPException(detail='Cannot kick the owner of the room', status_code=403)
    
    if username_to_kick not in room.users:
        raise HTTPException(detail='The person attempted to kick is not inside the room', status_code=403)
    
    room.users.remove(username_to_kick)
    del room.user_provided_labels[username_to_kick]
    if room.is_finished:
        del room.user_results[username_to_kick]
    del user2room[username_to_kick]
    rooms[room_name] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )
    
#   ,-.,--. ,--.        ,-.  ,--.                ,--.        ,------.                          
#  / .'|  | |  |,--,--, '. \ |  |    ,---.  ,---.|  |,-.     |  .--. ' ,---.  ,---. ,--,--,--. 
# |  | |  | |  ||      \ |  ||  |   | .-. || .--'|     /     |  '--'.'| .-. || .-. ||        | 
# |  | '  '-'  '|  ||  | |  ||  '--.' '-' '\ `--.|  \  \     |  |\  \ ' '-' '' '-' '|  |  |  | 
#  \ '. `-----' `--''--'.' / `-----' `---'  `---'`--'`--'    `--' '--' `---'  `---' `--`--`--' 
#   `-'                 `-'    
    
def change_room_lock_state(data: BasicRequest, room_name: str, new_lock_state: bool):
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    room = rooms[room_name]
    
    if room.is_finished:
        raise HTTPException(detail='The room can no longer be changed', status_code=403)
    
    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != data.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)
    
    room.is_locked = new_lock_state
    rooms[room_name] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )
    
# lock room
@post("/rooms/{room_name:str}/lock")
async def lock_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_lock_state(data, room_name, True)
    
# lock room
@post("/rooms/{room_name:str}/unlock")
async def unlock_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_lock_state(data, room_name, False)

#                                  ,--.                                                                                
# ,--.  ,--.,--.   ,--.           /  /,------.                               ,--.    ,------.                          
# |  '--'  |`--' ,-|  | ,---.    /  / |  .--. ' ,---.,--.  ,--.,---.  ,--,--.|  |    |  .--. ' ,---.  ,---. ,--,--,--. 
# |  .--.  |,--.' .-. || .-. :  /  /  |  '--'.'| .-. :\  `'  /| .-. :' ,-.  ||  |    |  '--'.'| .-. || .-. ||        | 
# |  |  |  ||  |\ `-' |\   --. /  /   |  |\  \ \   --. \    / \   --.\ '-'  ||  |    |  |\  \ ' '-' '' '-' '|  |  |  | 
# `--'  `--'`--' `---'  `----'/  /    `--' '--' `----'  `--'   `----' `--`--'`--'    `--' '--' `---'  `---' `--`--`--' 
#                            `--' 

def change_room_hidden_state(data: BasicRequest, room_name: str, new_hidden_state: bool):
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    
    room = rooms[room_name]
    
    if room.is_finished:
        raise HTTPException(detail='The room can no longer be changed', status_code=403)
    
    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != data.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)
    
    room.is_hidden = new_hidden_state
    rooms[room_name] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )
    
# hide room
@post("/rooms/{room_name:str}/hide")
async def hide_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_hidden_state(data, room_name, True)
    
# reveal room
@post("/rooms/{room_name:str}/reveal")
async def reveal_room(data: BasicRequest, room_name: str) -> Response:
    return change_room_hidden_state(data, room_name, False)
    
#  ,-----.,--.,--.                 ,--.      ,------.            ,--.              ,--. ,--.       ,--.                  ,--. 
# '  .--./|  |`--' ,---. ,--,--, ,-'  '-.    |  .-.  \  ,--,--.,-'  '-. ,--,--.    |  | |  | ,---. |  | ,---.  ,--,--. ,-|  | 
# |  |    |  |,--.| .-. :|      \'-.  .-'    |  |  \  :' ,-.  |'-.  .-'' ,-.  |    |  | |  || .-. ||  || .-. |' ,-.  |' .-. | 
# '  '--'\|  ||  |\   --.|  ||  |  |  |      |  '--'  /\ '-'  |  |  |  \ '-'  |    '  '-'  '| '-' '|  |' '-' '\ '-'  |\ `-' | 
#  `-----'`--'`--' `----'`--''--'  `--'      `-------'  `--`--'  `--'   `--`--'     `-----' |  |-' `--' `---'  `--`--' `---'  
#                                                                                           `--' 
    
# post data to room
@post("/submit-data")
async def receive_data(data: DataSubmissionRequest) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    
    
    # data to pandas conversion
    connections[data.id].data = pickle.loads(base64.b64decode(data.data.encode()))
    connections[data.id].data_labels = data.data_labels
    
    return Response(
        media_type=MediaType.TEXT,
        content='Data received',
        status_code=200
        )
    
# ,------.                     ,--. ,-----. ,------.   
# |  .--. ',--.,--.,--,--,     |  |'  .-.  '|  .-.  \  
# |  '--'.'|  ||  ||      \    |  ||  | |  ||  |  \  : 
# |  |\  \ '  ''  '|  ||  |    |  |'  '-'  '|  '--'  / 
# `--' '--' `----' `--''--'    `--' `-----' `-------' 
    
def run_riod(data, alpha):
    users = []
    
    ro.r['source']('./scripts/aggregation.r')
    aggregate_ci_results_f = ro.globalenv['aggregate_ci_results']
    # Reading and processing data
    #df = pl.read_csv("./random-data-1.csv")
    with (ro.default_converter + pandas2ri.converter).context():
        lvs = []
        for user, df, labels in data:
            #converting it into r object for passing into r function|
            d = [('citestResults', ro.conversion.get_conversion().py2rpy(df)), ('labels', ro.StrVector(labels))]
            od = OrderedDict(d)
            lv = ro.ListVector(od)
            lvs.append(lv)
            users.append(user)
        result = aggregate_ci_results_f(lvs, alpha)

        g_pag_list = [x[1].tolist() for x in result['G_PAG_List'].items()]
        g_pag_labels = [list(x[1]) for x in result['G_PAG_Label_List'].items()]
        gi_pag_list = [x[1].tolist() for x in result['Gi_PAG_List'].items()]
        gi_pag_labels = [list(x[1]) for x in result['Gi_PAG_Label_List'].items()]
        return g_pag_list, g_pag_labels,  {u:r for u,r in zip(users, gi_pag_list)}, {u:l for u,l in zip(users, gi_pag_labels)}
    
@post("/rooms/{room_name:str}/run")
async def run_iod(data: IODExecutionRequest, room_name: str) -> Response:
    if not validate_user_request(data.id, data.username):
        raise HTTPException(detail='The provided identification is not recognized by the server', status_code=401)
    if room_name not in rooms:
        raise HTTPException(detail='The room does not exist', status_code=404)
    room = rooms[room_name]

    # this is safe because, usernames are unique by nature and validate_user_request verifies correctness of id-username match
    if room.owner_name != data.username:
        raise HTTPException(detail='You do not have sufficient authority in this room', status_code=403)
    
    room.is_processing = True
    room.is_locked = True
    room.is_hidden = True
    rooms[room_name] = room
    
    # gather data of all participants
    participant_data = []
    participant_data_labels = []
    participants = room.users
    for user in participants:
        conn = user2connection[user]
        participant_data.append(conn.data)
        participant_data_labels.append(conn.data_labels)
        
    try:
        result, result_labels, user_result, user_labels = run_riod(zip(participants, participant_data, participant_data_labels), alpha=data.alpha)
    except:
        room = rooms[room_name]
        room.is_processing = False
        room.is_locked = True
        room.is_hidden = True
        rooms[room_name] = room
        raise HTTPException(detail='Failed to execute IOD', status_code=500)
    
    room = rooms[room_name]
    for user in user_result:
        if user not in room.users:
            del user_result[user]
    
    room.result = result
    room.result_labels = result_labels
    room.user_results = user_result
    room.user_labels = user_labels
    room.is_processing = False
    room.is_finished = True
    rooms[room_name] = room
    
    return Response(
        media_type=MediaType.JSON,
        content=RoomDetailsDTO(room, data.username),
        status_code=200
        )

# ,------.                  ,--.                     ,---.           ,--.                  
# |  .--. ' ,---. ,--.,--.,-'  '-. ,---. ,--.--.    '   .-'  ,---. ,-'  '-.,--.,--. ,---.  
# |  '--'.'| .-. ||  ||  |'-.  .-'| .-. :|  .--'    `.  `-. | .-. :'-.  .-'|  ||  || .-. | 
# |  |\  \ ' '-' ''  ''  '  |  |  \   --.|  |       .-'    |\   --.  |  |  '  ''  '| '-' ' 
# `--' '--' `---'  `----'   `--'   `----'`--'       `-----'  `----'  `--'   `----' |  |-'  
#                                                                                  `--' 

app = Litestar([
    health_check,
    change_name,
    check_in,
    create_room,
    get_rooms,
    get_room,
    join_room,
    leave_room,
    kick_user_from_room,
    receive_data,
    lock_room,
    unlock_room,
    hide_room,
    reveal_room,
    run_iod
    ])