from dataclasses import dataclass
import time
import datetime
import abc
import uuid
from co6co.utils import log
from co6co.storage.Dict import CallbackDict
try:
    import ujson
except ImportError:
    import json as ujson


def get_request_container(request):
    """
    用于获取请求的容器对象
    """
    return request.ctx.__dict__ if hasattr(request, "ctx") else request


class SessionDict(CallbackDict):
    def __init__(self, initial=None, sid=None):
        def on_update(self):
            self.modified = True

        super().__init__(initial, on_update)

        self.sid = sid
        self.modified = False


@dataclass(init=False)
class session_option: 
    useHeader:bool
    expiry: int
    """会话的过期时间，单位为秒""" 
    prefix: str
    """存储会话数据时使用的键前缀""" 
    head_name: str
    """头部名称，可能用于自定义请求头或响应头"""
    cookie_name: str
    domain: str
    """domain 必须匹配当前域名或父域名"""
    httponly: bool
    sessioncookie: bool 
    samesite: str
    """用于设置 Cookie 的 SameSite 属性 samesite='Strict' - 跨站请求不会发送 cookie"""

    session_name: str
    """会话对象在请求容器中的名称""" 
    secure: bool
    """是否使用安全的 Cookie（仅通过 HTTPS 传输）"""
    path: str
    """Cookie 的路径属性，默认为 '/' 表示整个域名下有效"""

    @staticmethod
    def crate_use_header(): 
        option = session_option()
        option.useHeader=True
        option.expiry = 2592000
        option.head_name = "session"
        option.prefix = "session:"
        option.samesite = None
        option.session_name = "Session"
        option.secure = False
        option.domain=None
        return option

    @staticmethod
    def crate_use_cookie():
        option = session_option()
        option.useHeader=False
        option.domain=None
        option.expiry = 2592000
        option.httponly = True
        option.cookie_name = "session"
        option.prefix = "session:"
        option.sessioncookie = False
        option.samesite = None
        option.session_name = "Session"
        option.secure = False
        option.path = "/"
        return option


class IBaseSession(metaclass=abc.ABCMeta):
    # this flag show does this Interface need request/response middleware hooks

    def __init__(
        self,  option: session_option
    ): 
         self.option=option
         

    def _delete_cookie(self, request, response):
        req = get_request_container(request)
        response.cookies[self.option.cookie_name] = ""

        # We set expires/max-age even for session cookies to force expiration
        response.cookies[self.option.cookie_name]["expires"] = datetime.datetime.utcnow()
        response.cookies[self.option.cookie_name]["max-age"] = 0

        if self.option.path:
            response.cookies[self.option.cookie_name]["path"] = self.option.path

        if self.option.domain:
            response.cookies[self.option.cookie_name]["domain"] = self.option.domain

    @staticmethod
    def _calculate_expires(expiry):
        expires = time.time() + expiry
        return datetime.datetime.fromtimestamp(expires)

    def _set_cookie_props(self, request, response):
        req = get_request_container(request)
        response.cookies[self.option.cookie_name] = req[self.option.session_name].sid
        response.cookies[self.option.cookie_name]["httponly"] = self.option.httponly

        if self.option.path:
            response.cookies[self.option.cookie_name]["path"] = self.option.path

        # Set expires and max-age unless we are using session cookies
        if not self.option.sessioncookie:
            response.cookies[self.option.cookie_name]["expires"] = self._calculate_expires(
                self.option.expiry)
            response.cookies[self.option.cookie_name]["max-age"] = self.option.expiry

        if self.option.domain:
            response.cookies[self.option.cookie_name]["domain"] = self.option.domain

        if self.option.samesite is not None:
            response.cookies[self.option.cookie_name]["samesite"] = self.option.samesite

        if self.option.secure:
            response.cookies[self.option.cookie_name]["secure"] = True

    @abc.abstractmethod
    async def _get_value(self, prefix: str, sid: str):
        """
        Get value from datastore. Specific implementation for each datastore.

        Args:
            prefix:
                A prefix for the key, useful to namespace keys.
            sid:
                a uuid in hex string
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def _delete_key(self, key: str):
        """Delete key from datastore"""
        raise NotImplementedError

    @abc.abstractmethod
    async def _set_value(self, key: str, data: SessionDict):
        """Set value for datastore"""
        raise NotImplementedError

    def getSid(self, request):
        if self.option.useHeader:
            sid = request.headers.get(self.option.head_name)
        else:
            sid = request.cookies.get(self.option.cookie_name)
        return sid

    async def open(self, request) -> SessionDict:
        """
        Opens a session onto the request. Restores the client's session
        from the datastore if one exists.The session data will be available on
        `request.session`.
        Args:
            request (sanic.request.Request):
                The request, which a sessionwill be opened onto.

        Returns:
            SessionDict:
                the client's session data,
                attached as well to `request.session`.
        """

        sid = self. getSid(request)
        if not sid:
            sid = uuid.uuid4().hex
            session_dict = SessionDict(sid=sid)
        else:
            val = await self._get_value(self.option.prefix, sid)

            if val is not None:
                try:
                    data = ujson.loads(val)
                    session_dict = SessionDict(data, sid=sid)
                except (ValueError, TypeError):
                    # Corrupted data, create new session
                    session_dict = SessionDict(sid=sid)
            else:
                session_dict = SessionDict(sid=sid)

        # attach the session data to the request, return it for convenience
        req = get_request_container(request)
        req[self.option.session_name] = session_dict
        return session_dict

    async def save(self, request, response) -> None:
        """Saves the session to the datastore.

        Args:
            request (sanic.request.Request):
                The sanic request which has an attached session.
            response (sanic.response.Response):
                The Sanic response. Cookies with the appropriate expiration
                will be added onto this response.

        Returns:
            None
        """
        req = get_request_container(request)
        if self.option.session_name not in req:
            return

        key = self.option.prefix + req[self.option.session_name].sid
        if not req[self.option.session_name]:
            await self._delete_key(key)
            
            if not self.option.useHeader and req[self.option.session_name].modified:
                self._delete_cookie(request, response)
            return

        val = ujson.dumps(dict(req[self.option.session_name]))
        await self._set_value(key, val)
        if not self.option.useHeader:
            self._set_cookie_props(request, response)
		
