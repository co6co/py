
from co6co.task.eventDispatcher import EventDispatcherProcess, Event, EventType, EventHandler, EventDispatcher
import pytest
import time
from multiprocessing import Pipe, Process


class StartEventDispatcher(EventHandler):
    def __init__(self, *serviceTypes: str):
        if len(serviceTypes) == 0:
            serviceTypes = ["start_task"]
        self.serviceType = serviceTypes
        super().__init__()

    def handle(self, event: Event) -> Event:
        if self.is_supported(event.event_type):  # EventType.TASK
            if event.event_type == "start_task":
                # 模拟处理任务
                result = f"任务已完成: {event.data}"
                return Event(
                    event_type='start_result',
                    data=result,
                    source="test_dispatcher",
                    timestamp=time.time()
                )
            else:
                # 模拟处理任务
                result = f"结果已处理: {event.data}"
                print(result)
                return None
        return None

    @property
    def supported_events(self):
        return self.serviceType


def test_hander_class():
    handler = StartEventDispatcher('start_task', 'start_result')
    assert handler.name == "StartEventDispatcher"
    print(handler.key, handler.__module__)
    assert handler.key == f"{handler.__module__}.StartEventDispatcher"
    assert handler.is_supported("start_task") == True
    assert handler.is_supported("start_result") == True
    assert handler.is_supported("error") == False


test_hander_class()
