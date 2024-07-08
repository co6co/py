from sanic import Sanic
from sanic.request import Request
from sanic.response import redirect, text
from urllib.parse import urlencode, unquote, unquote_plus, parse_qs
import argparse


oAuth2app = Sanic("OAuth2Service")
app = Sanic("MyApp")


@oAuth2app.route("/oauth/authorize")
async def authorize(request: Request):
    querys = request.query_string
    s = unquote(querys)
    s2 = unquote_plus(querys)
    print("unquote", s, parse_qs(s))
    querys = parse_qs(s)
    url = querys["redirect_uri"][0]
    print("unquote_plus", s2)
    state = querys["state"][0]
    code = "1233333333333333"
    return redirect(f"{url}?code={code}&state={state}")


@app.route("/login")
async def login(request):
    # 这里设置你的OAuth2授权服务器的登录URL
    auth_url = "http://127.0.0.1:8100/oauth/authorize"
    params = {
        "client_id": "your_client_id",
        "redirect_uri": "http://localhost:8000/callback",
        "response_type": "code",
        "scope": "read write",
        "state": "random_state_string"
    }
    url = f"{auth_url}?{urlencode(params)}"
    return redirect(url)


@app.middleware("response")
async def log_response_headers(request: Request, response):
    if response.status == 302:
        print(f"{request.url}->Redirect headers:", response.headers)


@app.route("/callback")
async def callback(request):
    # 从查询字符串中获取code
    code = request.args.get("code")
    state = request.args.get("state")

    # 在这里处理code，例如换取access token
    # ...
    # 重定向到一个页面，显示授权结果或进一步操作
    return redirect("/authorized", headers={"code": code, "state": state})


@app.route("/authorized")
async def authorized(request: Request):
    return text("Authorization successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="audit service.")
    parser.add_argument('-s', '--isServer', default=False,
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.isServer:
        oAuth2app.run(host="0.0.0.0", port=8100)
    else:
        app.run(host="0.0.0.0", port=8000)
