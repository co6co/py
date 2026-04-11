from sanic import Sanic, Websocket
from sanic.response import html
from model.apphelp import read_file_content,get_file_path 

app = Sanic("MyApp")

@app.websocket("/ws")
async def handle_ws(request, ws: Websocket):
    # 关键：不要开线程！不要传 ws 到线程！
    while True:
        data = await ws.recv()
        await ws.send(f"echo: {data}")

# ----------------------
# 2. 测试页面（浏览器打开就能连 WS）
# ----------------------
@app.get("/")
async def index(request):
    index_path=get_file_path('testIndex.html')
    html_content =   read_file_content(index_path)
    return html(html_content) 
if __name__ == '__main__':
   # 运行服务器
    app.run(
        host="0.0.0.0",
        port=8800,
        debug=True,
        access_log=True,
        auto_reload=True
    )