from litestar import Litestar, get, post


@get("/health-check")
async def health_check() -> str:
    return {'available': True}

# List all available rooms
@get("/rooms")
async def get_rooms() -> str:
    return {'available': True}

# Get information about room, if in room
@get("/rooms/{room_id:str}")
async def get_rooms(room_id: str) -> str:
    return {'available': True}

# make new room, return id of room
@post("/rooms")
async def create_room() -> str:
    return {'available': True}


app = Litestar([health_check])