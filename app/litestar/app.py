from litestar import Litestar, MediaType, get, post

from litestar.contrib.htmx.request import HTMXRequest
from litestar.contrib.htmx.response import HTMXTemplate
from litestar.response import Template

from litestar.contrib.jinja import JinjaTemplateEngine
from litestar.template.config import TemplateConfig

from pathlib import Path

import polars as pl


@post(path="/upload-data", media_type=MediaType.TEXT)
async def upload_data(request: HTMXRequest) -> str:
    data = await request.form()
    data = await data['file'].read()
    data = data.decode('utf-8')
    
    data = data.split('\n')
    data = [d.split(',') for d in data]
    header = data[0]
    content = data[1:]
    data = {}
    for i, h in enumerate(header):
        data[h] = [c[i] for c in content]
        
    df = pl.from_dict(data)
    
    # TODO: Store df in global state. Maybe show df to client
    # TODO: Improve htmx website
    
    return HTMXTemplate(template_name="main.html", push_url="/")

@get(path="/")
def get_base(request: HTMXRequest) -> Template:
    htmx = request.htmx  #  if true will return HTMXDetails class object
    if htmx:
        print(htmx.current_url)
    # OR
    if request.htmx:
        print(request.htmx.current_url)
    return HTMXTemplate(template_name="main.html", push_url="/")


app = Litestar(
    route_handlers=[get_base, upload_data],
    debug=True,
    request_class=HTMXRequest,
    template_config=TemplateConfig(
        directory=Path("./templates"),
        engine=JinjaTemplateEngine,
    ),
)