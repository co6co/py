from .xxx import IBaseSession
class MemcacheSessionInterface(IBaseSession):
    def __init__(self):
        pass
    def _dump_registry(cls, file = None):
        return super()._dump_registry(file)
