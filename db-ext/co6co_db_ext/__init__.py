# -*- coding:utf-8 -*-

'''
from .db_session import  db_service,connectSetting
from .actuator import Actuator
from .session import session_context, transactional
from .db_utils import db_tools,DbCallable, InsertCallable ,QueryOneCallable, UpdateOneCallable,QueryListCallable,QueryPagedCallable,QueryPagedByFilterCallable


__name__ = "co6co_db_ext"  # 显式定义
def __getattr__(name):
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(__all__ + ['__all__', '__name__', '__file__'])
__all__ = [
    "db_service",
    'connectSetting',
    "Actuator",
    "session_context",
    "transactional",
    "db_tools",
    "DbCallable",
    "InsertCallable",
    "UpdateCallable",
    "DeleteCallable",
    "QueryCallable",
    "QueryAllCallable",
    "QueryOneCallable",
    "UpdateOneCallable",
    "QueryListCallable",
    "QueryPagedCallable",
    "QueryPagedByFilterCallable",
] 
'''
 
import pkgutil
import importlib

def auto_import_models(*package_names):
    """
    自动导入 models所在的 BasePO
    该方法不适合大项目
    """
    for package_name in package_names: 
        package = importlib.import_module(package_name)
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package_name}.{module_name}")
        
__version_info = (0, 1,0)
__version__ = ".".join([str(x) for x in __version_info])
