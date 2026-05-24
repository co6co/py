from .base import IBaseSession, session_option

try:
    import aioredis
except ImportError:
    aioredis = None


class AIORedisSessionImp(IBaseSession):
    def __init__(
        self,
        redis,
        options: session_option = None
    ):
        """Initializes a session interface backed by Redis.

        Args:
            redis (Callable):
                aioredis connection or connection pool instance.
            domain (str, optional):
                Optional domain which will be attached to the cookie.
            expiry (int, optional):
                Seconds until the session should expire.
            httponly (bool, optional):
                Adds the `httponly` flag to the session cookie.
            cookie_name (str, optional):
                Name used for the client cookie.
            prefix (str, optional):
                Memcache keys will take the format of `prefix+session_id`;
                specify the prefix here.
            sessioncookie (bool, optional):
                Specifies if the sent cookie should be a 'session cookie', i.e
                no Expires or Max-age headers are included. Expiry is still
                fully tracked on the server side. Default setting is False.
            samesite (str, optional):
                Will prevent the cookie from being sent by the browser to the target
                site in all cross-site browsing context, even when following a regular link.
                One of ('lax', 'strict')
                Default: None
            session_name (str, optional):
                Name of the session that will be accessible through the request.
                e.g. If ``session_name`` is ``alt_session``, it should be
                accessed like that: ``request.ctx.alt_session``
                e.g. And if ``session_name`` is left to default, it should be
                accessed like that: ``request.ctx.session``
                Default: 'session'
            secure (bool, optional):
                Adds the `Secure` flag to the session cookie.
        """
        if aioredis is None:
            raise RuntimeError(
                "Please install aioredis: pip install sanic_session[aioredis]")

        self.redis = redis

        super().__init__(
            options
        )

    async def _get_value(self, prefix, sid):
        return await self.redis.get(self.option.prefix + sid)

    async def _delete_key(self, key):
        await self.redis.delete(key)

    async def _set_value(self, key, data):
        await self.redis.setex(key, self.option.expiry, data)
