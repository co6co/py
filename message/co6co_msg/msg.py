
from dataclasses import dataclass, field, asdict
from typing import Any, Dict,  Optional, Type, TypeVar
from datetime import datetime
import json
import uuid
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseMessage')

class MessageType(Enum):
    """预定义的消息类型枚举"""
    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"
    RESPONSE = "response"


@dataclass
class BaseMessage:
    """消息基类，所有消息都应继承此类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.EVENT
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """从字典创建消息实例"""
        # 处理枚举类型
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = MessageType(data['type'])
        
        # 处理 datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典"""
        result = asdict(self)
        # 转换枚举为字符串
        result['type'] = self.type.value
        # 转换 datetime 为 ISO 格式
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self) -> str:
        """将消息转换为 JSON 字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """从 JSON 字符串创建消息实例"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CommandMessage(BaseMessage):
    """命令消息基类"""
    type: MessageType = MessageType.COMMAND
    command_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventMessage(BaseMessage):
    """事件消息基类"""
    type: MessageType = MessageType.EVENT
    event_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMessage(BaseMessage):
    """查询消息基类"""
    type: MessageType = MessageType.QUERY
    query_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseMessage(BaseMessage):
    """响应消息基类"""
    type: MessageType = MessageType.RESPONSE
    success: bool = True
    data: Any = None
    error: Optional[str] = None