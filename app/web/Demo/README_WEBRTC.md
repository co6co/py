# WebRTC 信令服务器使用说明

## 功能概述

本 WebRTC 系统包含以下主要功能:

### 客户端功能 (webrtc_client.html)
1. **连接管理**
   - 连接/断开信令服务器
   - 自动注册到指定房间
   - 实时显示连接状态 (ICE 状态、信令状态、连接状态)

2. **用户管理**
   - 显示房间中的在线用户列表
   - 实时通知用户加入/离开
   - 支持点击呼叫特定用户

3. **视频通话**
   - 支持摄像头和屏幕共享两种视频源
   - 支持音频开关控制
   - 支持视频开关控制
   - 实时显示通话统计数据 (带宽、帧率、包数量、RTT)

4. **通知系统**
   - 弹出通知显示重要事件
   - 详细的事件日志记录

### 服务器端功能 (webrtc_server_for_http.py)
1. **信令服务**
   - WebSocket 信令服务器
   - 支持房间管理
   - 用户注册和发现
   - 消息转发 (Offer/Answer/ICE Candidate)

2. **房间管理**
   - 多房间支持
   - 自动创建房间
   - 房间状态跟踪

3. **监控和日志**
   - HTTP 状态端点：`/status`
   - 房间信息端点：`/api/rooms`
   - 详细的日志记录 (控制台 + 文件)

## 启动说明

### 1. 启动信令服务器

```bash
python webrtc_server_for_http.py
```

服务器将在以下地址运行:
- Web 界面：http://localhost:8765
- WebSocket 端点：ws://localhost:8765/ws
- 状态监控：http://localhost:8765/status
- 房间信息：http://localhost:8765/api/rooms

### 2. 访问客户端

在浏览器中打开: http://localhost:8765/webrtc_client.html

### 3. 使用流程

1. **连接服务器**: 点击"连接信令服务器"按钮
2. **查看在线用户**: 在右侧面板查看当前房间中的用户
3. **发起呼叫**: 
   - 方法 1: 点击用户列表中的"呼叫"按钮
   - 方法 2: 点击"开始通话"按钮 (自动呼叫第一个用户)
4. **接听呼叫**: 当收到呼叫请求时，点击"接听通话"按钮
5. **控制音视频**: 使用视频窗口上的按钮控制音频/视频开关
6. **结束通话**: 点击"挂断通话"按钮

## 配置说明

### 客户端配置

点击"设置"按钮可以配置:
- **信令服务器地址**: WebSocket 服务器地址 (默认：ws://localhost:8765/ws)
- **房间号**: 要加入的房间名称 (默认：default-room)
- **用户 ID**: 自己的标识 (不填写将自动生成)
- **视频源**: 摄像头 或 屏幕共享
- **音频源**: 麦克风 或 无音频

配置会自动保存到浏览器本地存储。

### 服务器端配置

服务器配置通过 `model/apphelp.py` 中的 `get_config()` 获取:
- **port**: 服务器端口 (默认：8765)

## API 文档

### WebSocket 消息格式

所有消息均为 JSON 格式:

#### 客户端 -> 服务器

1. **注册**
```json
{
    "type": "register",
    "peerId": "user_123",
    "roomId": "room_1"
}
```

2. **获取用户列表**
```json
{
    "type": "get-peer-list",
    "roomId": "room_1"
}
```

3. **呼叫请求**
```json
{
    "type": "call-request",
    "from": "user_1",
    "to": "user_2"
}
```

4. **SDP Offer**
```json
{
    "type": "offer",
    "sdp": "...",
    "from": "user_1",
    "to": "user_2"
}
```

5. **SDP Answer**
```json
{
    "type": "answer",
    "sdp": "...",
    "from": "user_2",
    "to": "user_1"
}
```

6. **ICE Candidate**
```json
{
    "type": "ice-candidate",
    "candidate": {...},
    "from": "user_1",
    "to": "user_2"
}
```

7. **挂断**
```json
{
    "type": "hangup",
    "from": "user_1",
    "to": "user_2"
}
```

#### 服务器 -> 客户端

1. **欢迎消息**
```json
{
    "type": "welcome",
    "peerId": "user_123",
    "roomId": "room_1"
}
```

2. **用户列表**
```json
{
    "type": "peer-list",
    "peers": ["user_1", "user_2"]
}
```

3. **用户加入**
```json
{
    "type": "peer-joined",
    "peerId": "user_3"
}
```

4. **用户离开**
```json
{
    "type": "peer-left",
    "peerId": "user_3"
}
```

5. **呼叫请求**
```json
{
    "type": "call-request",
    "from": "user_2"
}
```

6. **呼叫已接受**
```json
{
    "type": "call-accepted",
    "from": "user_2"
}
```

### HTTP 端点

#### GET /status
返回服务器状态:
```json
{
    "status": "running",
    "timestamp": "2026-04-27T10:30:00",
    "clients_count": 5,
    "rooms_count": 2,
    "rooms": {
        "room_1": {
            "peer_count": 3,
            "peers": ["user_1", "user_2", "user_3"]
        },
        "room_2": {
            "peer_count": 2,
            "peers": ["user_4", "user_5"]
        }
    }
}
```

#### GET /api/rooms
返回所有房间信息:
```json
{
    "total_rooms": 2,
    "rooms": [
        {
            "room_id": "room_1",
            "peer_count": 3,
            "peers": ["user_1", "user_2", "user_3"]
        },
        {
            "room_id": "room_2",
            "peer_count": 2,
            "peers": ["user_4", "user_5"]
        }
    ]
}
```

## 日志说明

### 服务器日志

服务器日志同时输出到:
- 控制台
- 文件：`webrtc_server.log`

日志级别: INFO
日志格式：`时间 - 级别 - 消息`

### 客户端日志

客户端日志显示在页面右下角的"事件日志"面板中，包含:
- 连接状态变化
- 消息收发记录
- 错误信息
- ICE 状态变化
- 通话统计信息

## 故障排除

### 常见问题

1. **无法连接服务器**
   - 检查服务器是否启动
   - 检查端口是否被占用
   - 检查防火墙设置

2. **无法获取摄像头/麦克风**
   - 检查浏览器权限设置
   - 确保使用 HTTPS 或 localhost
   - 检查设备是否被其他程序占用

3. **通话建立失败**
   - 检查 ICE 服务器配置
   - 检查网络连接
   - 查看浏览器控制台错误信息

4. **视频卡顿**
   - 检查网络带宽
   - 降低视频分辨率或帧率
   - 检查 CPU/内存使用情况

## 技术栈

### 客户端
- HTML5
- CSS3
- JavaScript (ES6+)
- WebRTC API
- WebSocket API

### 服务器端
- Python 3.7+
- aiohttp (WebSocket 服务器)
- asyncio (异步 IO)

## 安全注意事项

1. **生产环境部署**
   - 使用 HTTPS/WSS
   - 添加用户认证机制
   - 实现速率限制
   - 配置防火墙规则

2. **隐私保护**
   - 不记录媒体内容
   - 仅转发信令消息
   - 实现房间访问控制

3. **资源管理**
   - 监控服务器负载
   - 限制房间大小
   - 实现超时断开机制

## 未来改进

1. **功能增强**
   - 支持多人会议
   - 添加文字聊天功能
   - 支持文件传输
   - 实现会议录制

2. **性能优化**
   - 添加 TURN 服务器支持
   - 实现媒体质量自适应
   - 优化 ICE 候选收集

3. **用户体验**
   - 改进 UI/UX
   - 添加音效提示
   - 实现虚拟背景
   - 支持美颜滤镜
