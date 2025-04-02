
from abc import ABC, abstractmethod
# 定义接口


class ICustomTask(ABC):
    name = "抽象任务接口"
    code = "ICustomTask"

    @abstractmethod
    def main(self):
        pass
