from __future__ import annotations
from nats.aio.client import Client as NATS
from typing import  Dict, List, Optional, Type,  Callable, Awaitable 
import json 
import asyncio 
import logging
from .msg import BaseMessage, ResponseMessage

logger = logging.getLogger(__name__)

class NATService:
    """NATS 操作辅助类"""
    
    def __init__(self, nats_url: str = "nats://localhost:4222", client_name: str = "python-client"):
        self.nats_url = nats_url
        self.client_name = client_name
        self.nc: Optional[NATS] = None
        self.subscriptions: Dict[str, List[int]] = {}
        
    async def connect(self):
        """连接到 NATS 服务器"""
        if self.nc is None or self.nc.is_closed:
            self.nc = NATS()
            await self.nc.connect(
                servers=[self.nats_url],
                name=self.client_name,
                reconnect_time_wait=5,
                max_reconnect_attempts=-1,
            )
            logger.info(f"Connected to NATS at {self.nats_url}")
    
    async def disconnect(self):
        """断开连接"""
        if self.nc and not self.nc.is_closed:
            await self.nc.drain()
            await self.nc.close()
            logger.info("Disconnected from NATS")
    
    async def publish(
        self, 
        subject: str, 
        message: BaseMessage,
        reply_to: Optional[str] = None
    ):
        """发布消息"""
        await self.connect()
        
        payload = message.to_json().encode('utf-8')
        
        try:
            await self.nc.publish(subject, payload, reply=reply_to)
            logger.debug(f"Published message to {subject}: {message.id}")
        except Exception as e:
            logger.error(f"Failed to publish message to {subject}: {e}")
            raise
    
    async def request(
        self, 
        subject: str, 
        message: BaseMessage,
        timeout: float = 2.0
    ) -> ResponseMessage:
        """发送请求并等待响应"""
        await self.connect()
        
        payload = message.to_json().encode('utf-8')
        
        try:
            response = await self.nc.request(subject, payload, timeout=timeout)
            return ResponseMessage.from_json(response.data.decode('utf-8'))
        except asyncio.TimeoutError:
            logger.warning(f"Request to {subject} timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Request to {subject} failed: {e}")
            raise
    
    async def subscribe(
        self,
        subject: str,
        callback: Callable[[BaseMessage,NATService], Awaitable[None]],
        message_type: Type[BaseMessage] = BaseMessage,
        queue_group: Optional[str] = None
    ) -> int:
        """订阅主题
        
        Args:
            subject: 订阅的主题
            callback: 消息处理函数，接收消息对象
            message_type: 期望的消息类型，用于反序列化
            queue_group: 队列组名，用于负载均衡
            
        Returns:
            subscription_id: 订阅ID，可用于取消订阅
        """
        await self.connect()
        
        async def message_handler(msg):
            try:
                # 解析消息
                message_data = msg.data.decode('utf-8')
                message = message_type.from_json(message_data)
                
                # 添加 NATS 元数据
                message.metadata.update({
                    'nats_subject': msg.subject,
                    'nats_reply': msg.reply,
                    'nats_headers': dict(msg.headers) if msg.headers else {}
                })
                
                logger.debug(f"Received message on {msg.subject}: {message.id}")
                
                # 调用用户回调
                await callback(message,self)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from {msg.subject}: {e}")
            except Exception as e:
                logger.error(f"Error processing message from {msg.subject}: {e}")
        
        # 订阅主题
        if queue_group:
            sid = await self.nc.subscribe(
                subject, 
                cb=message_handler,
                queue=queue_group
            )
        else:
            sid = await self.nc.subscribe(
                subject, 
                cb=message_handler
            )
        
        # 记录订阅
        if subject not in self.subscriptions:
            self.subscriptions[subject] = []
        self.subscriptions[subject].append(sid)
        
        logger.info(f"Subscribed to {subject} (sid: {sid})")
        return sid
    
    async def unsubscribe(self, subject: str, sid: Optional[int] = None):
        """取消订阅"""
        if subject not in self.subscriptions:
            return
        
        if sid is None:
            # 取消该主题的所有订阅
            for sub_id in self.subscriptions[subject]:
                await self.nc.unsubscribe(sub_id)
            del self.subscriptions[subject]
            logger.info(f"Unsubscribed all from {subject}")
        else:
            # 取消特定订阅
            if sid in self.subscriptions[subject]:
                await self.nc.unsubscribe(sid)
                self.subscriptions[subject].remove(sid)
                logger.info(f"Unsubscribed sid {sid} from {subject}")
    
    async def create_stream(self, stream_name: str, subjects: List[str]):
        """创建 JetStream 流（如果需要持久化）"""
        await self.connect()
        
        try:
            js = self.nc.jetstream()
            await js.add_stream(name=stream_name, subjects=subjects)
            logger.info(f"Created stream {stream_name} with subjects {subjects}")
        except Exception as e:
            logger.error(f"Failed to create stream {stream_name}: {e}")
            raise
