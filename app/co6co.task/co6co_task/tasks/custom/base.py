
from abc import ABC, abstractmethod
from co6co_sanic_ext import sanics
# 定义接口


class ICustomTask(ABC):
    name = "抽象任务接口"
    code = "ICustomTask"

    def __init__(self, worker: sanics.Worker = None):

        super().__init__()
        self.worker = worker

    @abstractmethod
    def main(self):
        pass
