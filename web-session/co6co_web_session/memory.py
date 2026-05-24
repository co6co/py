from .base import IBaseSession, session_option
from co6co.storage.Dict import ExpiringDict


class MemorySessionImp(IBaseSession):
    def __init__(
        self,
        option: session_option = None
    ):
        if option == None:
            option = session_option.crate_use_header()
        super().__init__(option=option)
        self.session_store = ExpiringDict()

    async def _get_value(self, prefix, sid):
        return self.session_store.get(self.option.prefix + sid)

    async def _delete_key(self, key):
        if key in self.session_store:
            self.session_store.delete(key)

    async def _set_value(self, key, data):
        self.session_store.set(key, data, self.option.expiry)
