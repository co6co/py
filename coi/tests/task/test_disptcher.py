from co6co.task.eventDispatcher import EventDispatcherProcess, Event, EventType, EventHandler, EventDispatcher
import pytest
import time
from multiprocessing import Pipe, Process


@pytest.fixture
def pipeData():
    parent_conn, child_conn = Pipe()
    return parent_conn, child_conn


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
    # assert handler.key == f"{handler.__module__}.StartEventDispatcher"
    assert handler.is_supported("start_task") == True
    assert handler.is_supported("start_result") == True
    assert handler.is_supported("error") == False


def test_event_dispatcher():
    s = EventDispatcher()
    e1 = StartEventDispatcher("start_task")
    e2 = StartEventDispatcher("start_result")
    e3 = StartEventDispatcher("error")
    e4 = StartEventDispatcher("error")

    s.register_handler(e1)
    s.register_handler(e2)
    s.register_handler(e3)
    s.register_handler(e4)
    s.unregister_handler(e2)
    assert e2.supported_events[0] not in s.handlers
    assert e1.supported_events[0] in s.handlers and e3.supported_events[0] in s.handlers

    s.clear_handlers('error')
    print(s.handlers)
    assert 'error' not in s.handlers and 'start_task' in s.handlers
    s.clear_handlers()
    assert 'start_task' not in s.handlers


def test_event_dispatcher(pipeData):
    # parent_conn, child_conn = pipeData
    parent_conn, child_conn = Pipe()
    dispatcher = EventDispatcherProcess(parent_conn, 'parent')
    dispatcher.resgister_handler(StartEventDispatcher)
    dispatcher.start()

    dispatcher2 = EventDispatcherProcess(child_conn, 'child')
    dispatcher2.resgister_handler(StartEventDispatcher)
    dispatcher2.start()

    # 发送一个任务事件
    event = {
        "event_type": "start_task",
        "data": {"task_id": 123, "task_data": "Hello, World!"},
        "source": "test_client",
        "timestamp": 1234567890.0
    }
    event = Event.from_dict(event)
    dispatcher.send(event)
    print("[Test] 事件已发送")

    # 等待处理结果
    import time
    time.sleep(1)  # 等待事件处理
    parent_conn.close()
    child_conn.close()
    dispatcher.stop()
    dispatcher2.stop()
