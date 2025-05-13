from co6co_sanic_ext import sanics
from sanic import response


def init(app):
    @app.get("/")
    async def index(request):
        return response.html("Hello World!!")


if __name__ == '__main__':
    sanics.startApp({"web_setting": {"port": 30000}}, init)
