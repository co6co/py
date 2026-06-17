from nats.aio.client import Client as NATS
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Awaitable
from datetime import datetime
import json
import uuid
import asyncio
from enum import Enum
import logging
from co6co_msg.msg import EventMessage,BaseMessage, CommandMessage, QueryMessage,ResponseMessage
from co6co_msg.services import NATService


# ==================== 使用示例 ==================== 
# 1. 定义具体的消息类型
@dataclass
class UserCreatedEvent(EventMessage):
    """用户创建事件"""
    event_name: str = "user.created"
    
    # 自定义字段
    user_id: str = ""
    username: str = ""
    email: str = ""


@dataclass
class CreateUserCommand(CommandMessage):
    """创建用户命令"""
    command_name: str = "create_user"
    
    # 自定义字段
    username: str = ""
    email: str = ""
    password: str = ""


@dataclass
class GetUserQuery(QueryMessage):
    """获取用户查询"""
    query_name: str = "get_user"
    
    # 自定义字段
    user_id: str = ""


# 2. 使用示例
async def example_usage():
    # 初始化 NATS 助手
    nats_helper = NATService(
        nats_url="nats://localhost:4222",
        client_name="example-service"
    )
    
    # 连接
    await nats_helper.connect()
    
    # 创建 JetStream 流（可选）
    await nats_helper.create_stream(
        stream_name="USER_EVENTS",
        subjects=["user.*"]
    )
    
    # 订阅主题
    async def handle_user_created(event: UserCreatedEvent):
        print(f"User created: {event.username} ({event.email})")
        # 业务逻辑...
    
    await nats_helper.subscribe(
        subject="user.created",
        callback=handle_user_created,
        message_type=UserCreatedEvent,
        queue_group="user-service-group"  # 队列组，实现负载均衡
    )
    
    # 发布事件
    event = UserCreatedEvent(
        source="user-service",
        user_id="123",
        username="john_doe",
        email="john@example.com",
        metadata={"version": "1.0"}
    )
    await nats_helper.publish("user.created", event)
    
    # 发送命令
    command = CreateUserCommand(
        source="api-gateway",
        username="jane_doe",
        email="jane@example.com",
        password="secure_password"
    )
    await nats_helper.publish("commands.create_user", command)
    
    # 请求-响应模式
    async def handle_get_user(query: GetUserQuery) -> ResponseMessage:
        # 模拟数据库查询
        user_data = {
            "id": query.user_id,
            "username": "john_doe",
            "email": "john@example.com"
        }
        return ResponseMessage(
            correlation_id=query.id,
            success=True,
            data=user_data
        )
    
    # 订阅查询
    await nats_helper.subscribe(
        subject="queries.get_user",
        callback=lambda msg: asyncio.create_task(handle_query_request(msg)),
        message_type=GetUserQuery
    )
    
    async def handle_query_request(query: GetUserQuery):
        response = await handle_get_user(query)
        # 发送响应（如果有 reply_to）
        if hasattr(query.metadata, 'nats_reply') and query.metadata['nats_reply']:
            await nats_helper.publish(
                query.metadata['nats_reply'],
                response
            )
    
    # 发送请求并等待响应
    query = GetUserQuery(
        source="api-gateway",
        user_id="123"
    )
    try:
        response = await nats_helper.request(
            subject="queries.get_user",
            message=query,
            timeout=5.0
        )
        print(f"Got response: {response.data}")
    except asyncio.TimeoutError:
        print("Request timed out")
    
    # 清理
    await nats_helper.disconnect()


# 3. 便捷函数装饰器
def nats_subscriber(
    helper: NATService,
    subject: str,
    message_type: Type[BaseMessage] = BaseMessage,
    queue_group: Optional[str] = None
):
    """订阅装饰器，简化订阅定义"""
    def decorator(func: Callable[[BaseMessage], Awaitable[None]]):
        async def wrapper():
            await helper.subscribe(
                subject=subject,
                callback=func,
                message_type=message_type,
                queue_group=queue_group
            )
        wrapper.subject = subject
        wrapper.message_type = message_type
        return wrapper
    return decorator


# 使用装饰器的示例
async def example_with_decorator():
    nats_helper = NATService()
    await nats_helper.connect()
    
    @nats_subscriber(nats_helper, "user.created", UserCreatedEvent, "user-group")
    async def process_user_created(event: UserCreatedEvent):
        print(f"Processing user creation: {event.username}")
        # 业务逻辑...
    
    # 启动订阅
    await process_user_created()
    
    # 保持运行
    try:
        await asyncio.Future()  # 永久运行
    except asyncio.CancelledError:
        await nats_helper.disconnect()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())