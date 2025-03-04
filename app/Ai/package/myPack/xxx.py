import requests
from co6co.utils import log
import abc
def solution(url):
    r = requests.get(url)
    log.warn(r.text)
    return r.text 
class IBaseSession(metaclass=abc.ABCMeta):
    def __init__():
        pass
