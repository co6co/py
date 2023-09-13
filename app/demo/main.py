
import co6co.utils.log as log
import Crypto
import co6co
import co6co.utils.http as http
import json

import socket
import socks


log.start_mark("debug")
log.succ(f"co6co 版本：{co6co.__version__}" ) 
log.warn(dir( log))
print(dir(Crypto))
log.start_mark("debug.")

log.start_mark("set socks proxy test") 
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 9666)
socket.socket = socks.socksocket

log.log("获取当前IP:")
response=http.get("http://ip-api.com/json")
if response.status_code==200:
    log.succ(response.text)
    decoder = json.JSONDecoder()
    result=decoder.decode(response.text)
    log.succ(f"{type(response.text)} ->{type(result)}:\r\n{result}")
else: log.err("request error")
log.start_mark("set socks proxy test")