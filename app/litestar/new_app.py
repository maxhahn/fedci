from litestar import Litestar, get, Response, MediaType
from .helpers import cleanse_inactive_users
from .env import last_cleanse_time
import datetime

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


Litestar([
    health_check
])
