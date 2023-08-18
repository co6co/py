from co6co_clash.nodes import *
from co6co_clash.nodeLink import  _safe_decode
import  base64
from urllib import parse

if __name__ =="__main__":
    import co6co.utils.http as http
    url="http://127.0.0.1/res/sub2.txt"
    response = http.get(url)
    li=parser_content(response.text) 
    ss="MTUwLjEwNy40Ni4yMTo4MDgzOm9yaWdpbjphZXMtMjU2LWNmYjp0bHMxLjJfdGlja2V0X2F1dGg6YVVaeGJucFRjMk5PLz9vYmZzcGFyYW09JnJlbWFya3M9OEolMkJIcmZDZmg3RG5tYjNscTVZdE5EazAmcHJvdG9wYXJhbT0="
   
    s="8J%2BHrfCfh7Dnmb3lq5YtNDk0" 
    _safe_decode(s)
