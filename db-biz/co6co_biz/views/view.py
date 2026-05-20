from co6co_db_ext.actuator import Actuator
from co6co.data.result import Result, Page_Result
from co6co.utils.json_util import JSONEncoder


class AbsView:
    routePath="/api/v0"
    def __init__(self, actuator: Actuator) -> None:
        self._ctuator = actuator
        pass

    @property
    def actuator(self):
        return self._ctuator

    @property
    def current_user_id(self):
        raise ImportError()

    @property
    def current_user_name(self):
        raise ImportError

    def response_data(self, data: Result | Page_Result):
        result = JSONEncoder.dumps(data)
        raise
    def exist(self, isExist: bool = True, tableName="用户", name: str = "xxx"):
        if isExist:
            return self.response_data(Result.success(data=True, message=f"{tableName}'{name}'已存在。"))
        else:
            return self.response_data(Result.success(data=False, message=f"{tableName}'{name}'不已存在。"))
