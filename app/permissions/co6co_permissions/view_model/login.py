from typing import Optional
from co6co_sanic_ext.view_model import response_json
from co6co.data.result import Result

from sqlalchemy.sql import Select, Update
from co6co_db_ext.db_utils import db_tools 

from co6co.utils import log
from .base_view import AbsClsView
from ..services.utils import appHelper

from ..model.pos.right import UserPO
from .aop.login_log import loginLog, verifyCode
from ..services import getSecret, generatePageToken
from ..model.enum import user_state
from datetime import datetime
from ..services.configCache import get_user_config


class login_view(AbsClsView):
    routePath = "/login"

    @verifyCode
    @loginLog
    async def post(self):
        """
        登录
        """
        try:
            where = UserPO()
            isLock = -1  # 1 锁定 0 解锁
            session = self.db_session

            config = await get_user_config(self.request)
            failTimes = config.get("loginFail", 3)
            seconds = config.get("lockSeconds", 60)
            where.__dict__.update(self.json)
            status = [user_state.enabled.val, user_state.locked.val]
            select = Select(UserPO).filter(
                UserPO.userName.__eq__(where.userName), UserPO.state.in_(status)
            )
            user:Optional[UserPO] = await self.actuator.query_one_entity(select)
            _, sessionDict = self.get_Session(self.request)

            if user is not None:
                lockTime: datetime | None = user.lockTime
                if lockTime is None:
                    lockTime = datetime.now()
                lockTime = lockTime.timestamp()
                now = datetime.now().timestamp()
                waitingLogin: float = seconds - (now - lockTime)
                if (
                    user.state == user_state.locked.val
                    and lockTime
                    and waitingLogin > 0
                ):
                    return response_json(
                        Result.fail(
                            message=f"账号已锁定，{round(waitingLogin, 3)}s后再试!"
                        )
                    )

                if user.password == user.encrypt(where.password):
                    tokenData = await generatePageToken(
                        getSecret(self.request), user, userOpenId=user.userGroupId
                    )
                    # 让日志能获得用户信息
                    # log.warn("用户状态", user.state)
                    if user.state == user_state.locked.val:
                        isLock = 0
                    appHelper.set_current_user(self.request, user.jwt_data)
                    return response_json(
                        Result.success(data=tokenData, message="登录成功")
                    )
                else:
                    logingErrorCount = sessionDict.get("logingErrorCount", 0)
                    log.info("登录失败次数", logingErrorCount)
                    logingErrorCount += 1
                    sessionDict["logingErrorCount"] = logingErrorCount
                    if logingErrorCount == failTimes:
                        # 锁定
                        isLock = 1
                        sessionDict["logingErrorCount"] = 0
                        log.warn(
                            f"用户登录失败次数达到{logingErrorCount}次，账号已锁定。"
                        )
                    return response_json(
                        Result.fail(message="登录用户名或者密码不正确!")
                    )
            else:
                log.warn(f"未找到用户名[{where.userName}],。")
                return response_json(Result.fail(message="登录用户名或者密码不正确!"))
        except Exception as e:
            log.err(f"登录失败:{e}", e)
            return response_json(Result.fail(message="登录失败!"))
        finally:
            if isLock != -1:
                updateSml = (
                    Update(UserPO)
                    .filter(UserPO.id.__eq__(user.id))
                    .values(
                        {
                            UserPO.state: user_state.locked.val
                            if isLock == 1
                            else user_state.enabled.val,
                            UserPO.lockTime: datetime.now() if isLock == 1 else None,
                        }
                    )
                )
                await db_tools.execSQL(session, updateSml)
                await session.commit()
            pass
