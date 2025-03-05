
from .mem import MemcacheSessionInterface  # 假设你要导出的是 YourClass

__version_info = (0, 0, 1)
__version__ = ".".join([str(x) for x in __version_info])
__all__ = (
    "MemcacheSessionInterface",
)
