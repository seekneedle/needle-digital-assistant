import uvicorn
from utils.config import config
from fastapi import FastAPI
from server.assistant_router import store_router, ws_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html


app = FastAPI(docs_url=None)
# 包含路由
app.include_router(store_router)
app.include_router(ws_router)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.staticfile.org/swagger-ui/5.2.0/swagger-ui-bundle.min.js",  # 自定义JS URL
        swagger_css_url="https://cdn.staticfile.org/swagger-ui/5.2.0/swagger-ui.min.css"  # 自定义CSS URL
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=config['origins'],  # 允许的来源
    allow_credentials=True,  # 是否允许凭证传输，如cookies等敏感信息
    allow_methods=['*'],     # 允许的方法，默认为['GET']
    allow_headers=['*'],     # 允许的头部，默认为空列表
)


def start_server():
    ip = config['ip']
    port = config['port']
    uvicorn.run("server.server:app", host=ip, port=port, reload=False, workers=1)
