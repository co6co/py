import multiprocessing
import time
from multiprocessing import Process, Pipe
from multiprocessing.connection import PipeConnection
from typing import Dict, Any, Callable, Optional
from enum import Enum
import json
from dataclasses import dataclass, asdict
import threading
from abc import ABC, abstractmethod
from ..enums import Base_Enum
from typing import TypeVar, Optional, Type
from ..utils import log
import pickle ,struct



class EventType(Base_Enum):
    """事件类型"""
    TASK = "task", 0
    RESULT = "result", 1
    ERROR = "error", 2
    SHUTDOWN = "shutdown", 3
    HEARTBEAT = "heartbeat", 4

    @staticmethod
    def convert(keyOrVal: str | int):
        """根据键或值转换为枚举成员,如果失败则返回原始值"""
        result = EventType.key2enum(keyOrVal)
        if result is None:
            result = EventType.val2enum(keyOrVal)
        if result is None:
            return keyOrVal


@dataclass
class Event:
    """事件基类"""
    event_type: EventType | str | int
    data: Any
    source: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self):
        return {
            "event_type": self.event_type.key if isinstance(self.event_type, EventType) else self.event_type,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            event_type=EventType.convert(data["event_type"]),
            data=data["data"],
            source=data["source"],
            timestamp=data["timestamp"]
        )

    @staticmethod
    def create( event_type: EventType | str | int, sosurce: str, data: Any):
        """创建事件对象"""
        return Event(
            event_type=event_type,
            data=data,
            source=sosurce,
            timestamp=time.time()
        )


class EventHandler(ABC):
    """事件处理器基类"""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.process: EventDispatcherProcess = None

    def send(self, event: Event):
        """发送事件"""
        if self.process is not None:
            self.process.send(event)

    @abstractmethod
    def handle(self, event: Event) -> Optional[Event]:
        """处理事件，返回响应事件或None"""
        pass

    @property
    def key(self):
        cls = self.__class__
        return f"{cls.__module__}.{cls.__name__}" 

    @property
    @abstractmethod
    def supported_events(self):
        return [EventType.ERROR]

    def is_supported(self, event_type: EventType) -> bool:
        """检查是否支持处理该事件类型"""
        return event_type in self.supported_events

    def create_event(self, event_type: EventType | str | int, data: Any) -> Event:
        """创建事件对象"""
        return Event.create(event_type, self.name, data) 

T = TypeVar('T', bound='EventHandler')

class TaskHandler(EventHandler):
    """任务处理器"""

    def handle(self, event: Event) -> Optional[Event]:
        if self.is_supported(event.event_type):  # EventType.TASK
            print(f"[TaskHandler] 处理任务: {event.data}")
            # 模拟处理
            time.sleep(0.1)
            result = f"任务完成: {event.data}"

            return Event(
                event_type=EventType.RESULT,
                data=result,
                source=self.name,
                timestamp=time.time()
            )
        return None

    @property
    def supported_events(self):
        return [EventType.TASK]


class ErrorHandler(EventHandler):
    """错误处理器"""

    def handle(self, event: Event) -> Optional[Event]:
        if self.is_supported(event.event_type):  # EventType.ERROR
            print(f"[ErrorHandler] 处理错误: {event.data}")
            # 可以记录日志、发送报警等
            return Event(
                event_type=EventType.RESULT,
                data=f"错误已处理: {event.data}",
                source=self.name,
                timestamp=time.time()
            )
        return None

    @property
    def supported_events(self):
        return [EventType.ERROR]


class EventDispatcher:
    """事件分发器"""

    def __init__(self):
        self.handlers: Dict[EventType | str | int, list[EventHandler]] = {}
        self.name = self.__class__.__name__
    def exist(self,key:str):
        """检查事件处理器是否存在"""
        return self.get_handler(key) is not None
    def get_handler(self,key:str):
        """获取事件处理器"""
        for _,handlers in self.handlers.items():
            for handler in handlers:
                if handler.key == key:
                    return handler
        return None
    def register_handler(self,   handler: EventHandler):
        """注册事件处理器"""
        try:
            for event_type in handler.supported_events:
                if event_type in self.handlers:
                    self.handlers[event_type].append(handler)
                else:
                    self.handlers[event_type] = [handler]   
        except Exception as e:
            log.warn(f"注册事件处理器{handler.key}失败: {e}")
            pass
        

    def unregister_handler(self, event_type: EventType | str | int, handler: EventHandler) -> None:
        """移除事件处理器"""
        if event_type in self.handlers:
            try:
                self.handlers[event_type].remove(handler)
            except ValueError:
                pass
            # 如果列表为空，删除这个事件类型
            if not self.handlers[event_type]:
                del self.handlers[event_type]

    def unregister_handlers(self, handler: EventHandler) -> None:
        """移除所有事件处理器"""
        for event_type in handler.supported_events:
            self.unregister_handler(event_type, handler)

    def clear_handlers(self, event_type: EventType = None) -> None:
        """清除事件处理器

        Args:
            event_type: 如果为None，清除所有事件处理器
        """
        if event_type is None:
            self.handlers.clear()
        elif event_type in self.handlers:
            del self.handlers[event_type]

    def dispatch(self, event: Event) -> list[Event]:
        """分发事件给所有注册的处理器"""
        responses = []
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                try:
                    response = handler.handle(event)
                    if response:
                        responses.append(response)
                except Exception as e:
                    log.warn(f"事件处理器{handler.key}处理事件{event.data}失败: {e}")
                    #error_event = Event(
                    #    event_type=EventType.ERROR,
                    #    data=str(e),
                    #    source=self.name,
                    #    timestamp=time.time()
                    #)
                    #responses.append(error_event)
        else:
            log.warn(f"未注册事件类型: {event.event_type}")
        return responses


class EventDispatcherProcess:
    """事件分发器进程
    通过类注册的 EventHandler 不能动态管理，只能在进程启动前注册，进程启动后不能动态注册或移除处理器
    通过对象注册的 EventHandler 可以动态管理，进程启动后可以动态注册或移除处理器
    """

    def __init__(self,  conn: PipeConnection, worker_id: int | str = "worker"):
        self.conn = conn
        self.isQuit = False
        self._is_running = False
        self.worker_id = worker_id
        self.handler_classes = [] 
        self.dispatcher =  EventDispatcher()
        self.chuck_size=4096

    @property
    def is_running(self):
        return self._is_running

    def resgister_handler(self, *handler_claszes: Type[T]):
        """注册自定义事件处理器类"""
        if self.is_running:
            print(f"[Worker {self.worker_id}] 已经在运行，无法注册处理器")
            return
        self.handler_classes.extend(handler_claszes)

    def append_handler(self,   handler:  T):
        """注册单个自定义事件处理器类"""
        handler.process = self
        self.dispatcher.register_handler(handler)

    def remove_handler(self, event_type: EventType | str | int, handler:  T):
        """移除单个自定义事件处理器类"""

        self.dispatcher.unregister_handler(event_type, handler)
        handler.process = None

    def remove_handlers(self, handler:  T):
        """移除所有自定义事件处理器类"""
        self.dispatcher.unregister_handlers(handler)
        handler.process = None

    def clear_handlers(self):
        """清除所有通过对象注册的事件处理器"""
        if self.is_running:
            self.dispatcher.clear_handlers()

    def clear_handler_classes(self):
        """
        清除 以类注册的事件处理器
        注意：只能在进程启动前调用，进程启动后调用无效

        """
        if self.is_running:
            print(f"[Worker {self.worker_id}] 已经在运行，无法清除处理器")
            return
        self.handler_classes.clear()

    def stop(self):
        """
        发送关闭事件
        请在上层调用：PipeConnection.close()
        """
        self.isQuit = True
        # shutdown_event = Event(
        #    event_type=EventType.SHUTDOWN,
        #    data=None,
        #    source=f"main_process",
        #    timestamp=time.time()
        # )
        # self.conn.send(shutdown_event.to_dict())
        # time.sleep(0.5)  # 等待子进程处理关闭事件
        # self.conn.close() # 请在
        # self.thread.join()

    def send(self, event: Event):
        """发送事件到主进程"""
        try:
            # 这里可以实现发送事件到主进程的逻辑，例如通过管道或队列
            data=pickle.dumps(event)
            size = len(data)
            self.conn.send(struct.pack('>I', size))
            for i in range(0,size,self.chuck_size):
                self.conn.send(data[i:i+self.chuck_size])
        except Exception as e:
            log.err(f"发送事件到主进程失败: {e}",e)

    def _worker(self):
        # 创建事件分发器

        # 注册默认处理器
        #self.dispatcher .register_handler(TaskHandler())
        #self.dispatcher .register_handler(ErrorHandler())

        # 注册自定义处理器
        if self.handler_classes:
            for handler_class in self.handler_classes:
                handler: EventHandler = handler_class()
                handler.process = self

                # 根据处理器支持的事件类型注册
                if hasattr(handler, 'supported_events'):
                    for event_type in handler.supported_events:
                        self.dispatcher .register_handler(handler)

        
        # 心跳处理器
        '''
        class HeartbeatHandler(EventHandler):
            def handle(self, event: Event) -> Optional[Event]:
                if event.event_type == EventType.HEARTBEAT:
                    return Event(
                        event_type=EventType.HEARTBEAT,
                        data={"worker_id": worker_id, "status": "alive"},
                        source=f"worker_{worker_id}",
                        timestamp=time.time()
                    )
                return None

            @property
            def supported_events(self):
                return [EventType.HEARTBEAT]

        self.dispatcher.register_handler(HeartbeatHandler())
        '''
        # 主循环
        while True:
            try:
                if self.isQuit:
                    print(f"[Worker {self.worker_id}] 收到退出信号")
                    break
                data = self.conn.recv()
                size = struct.unpack('>I', data[:4])[0]
                data = data[4:]
                while len(data) < size:
                    data += self.conn.recv()
               
                event=pickle.loads(data)
                if isinstance(event,Dict):
                    #log.info(type(event),event)
                    event =Event.from_dict(event)
                if event.event_type == EventType.SHUTDOWN:
                    print(f"[Worker {self.worker_id}] 收到关闭事件")
                    break

                # 分发事件
                responses = self.dispatcher.dispatch(event)
                # 发送所有响应
                for response in responses: 
                    self.send(response.to_dict())
 
            except EOFError:
                print(f"[Worker {self.worker_id}] 连接中断")
                break
            except Exception as e: 
                log.err(f"[Worker {self.worker_id}] 错误: {e}",e)
                #error_event = Event(
                #    event_type=EventType.ERROR,
                #    data=str(e),
                #    source=f"worker_{self.worker_id}",
                #    timestamp=time.time()
                #)
                #self.send(error_event.to_dict())

        print(f"[Worker {self.worker_id}] 退出")
        # self.conn.close()

    def start(self):
        """工作进程 - 使用事件驱动架构"""
        if self._is_running:
            print(f"[Worker {self.worker_id}] 已经在运行")
            return
        print(f"[Worker {self.worker_id}] 启动")
        self.thread = threading.Thread(target=self._worker, name=f"worker_{self.worker_id}")  # , args=(conn,)
        self.thread.start()
        self._is_running = True
