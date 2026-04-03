from co6co.task.eventDispatcher import EventDispatcherProcess, Event, EventType, EventHandler, EventDispatcher
import pytest
import time
from multiprocessing import Pipe, Process


@pytest.fixture
def pipeData():
    parent_conn, child_conn = Pipe()
    return parent_conn, child_conn


class StartEventDispatcher(EventHandler):
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
        return ["start_task", "start_result"]


def test_hander_class():
    handler = StartEventDispatcher()
    assert handler.key == f"{handler.__module__}.StartEventDispatcher"
    assert handler.is_supported("start_task") == True
    assert handler.is_supported("start_result") == True
    assert handler.is_supported("error") == False


def test_test_event_dispatcher():
    s = EventDispatcher()
    e1 = StartEventDispatcher()
    e2 = StartEventDispatcher()
    e3 = StartEventDispatcher()

    s.register_handler("1", e1)
    s.register_handler("2", e2)
    s.register_handler("3", e3)
    s.register_handler("4", e3)
    s.unregister_handler("2", e2)
    assert "2" not in s.handlers and '1' in s.handlers and '3' in s.handlers
    s.clear_handlers('1')
    assert '1' not in s.handlers and '3' in s.handlers
    s.clear_handlers()
    assert '3' not in s.handlers


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
