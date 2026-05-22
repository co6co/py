from co6co_web_session import Session, IBaseSession, MemorySessionImp
from sanic.app import Sanic
import warnings


def init(app: Sanic, sessionImp: IBaseSession = MemorySessionImp()):
    """
    初始化 session
    应废弃，请使用使用 co6co_web_session.Session.mount_imp 初始化
    """
    warnings.warn("init session is deprecated, please co6co_web_session.Session.mount_imp", DeprecationWarning)
    session: Session = Session(app, sessionImp)
    return session
