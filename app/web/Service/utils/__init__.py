
from functools import wraps
from co6co.utils import log
from wechatpy import WeChatClient
from sanic.request import Request
from sanic.response import redirect, raw
from wechatpy import messages, events

from wechatpy.oauth import WeChatOAuth

import uuid
import inspect
from co6co.utils import log


def createUuid():
    return uuid.uuid4()
