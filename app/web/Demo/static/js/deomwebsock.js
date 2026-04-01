// static/js/websocket_client.js
class WebSocketClient {
    constructor() {
        this.ws = null;
        this.clientId = null;
        this.username = '用户' + Math.floor(Math.random() * 1000);
        this.roomId = 'room1';
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.messageCount = 0;
        this.connectedAt = null;
        this.pingInterval = null;
        this.lastPingTime = null;
        this.pingLatency = 0;
        
        // 状态跟踪
        this.stats = {
            messagesSent: 0,
            messagesReceived: 0,
            errors: 0,
            reconnects: 0
        };
        this.elements = {
            status: document.getElementById('status'),
            messages: document.getElementById('messages'),
            username: document.getElementById('username'),
            roomId: document.getElementById('roomId'),
            serverUrl: document.getElementById('serverUrl'),
            connectBtn: document.getElementById('connectBtn'),
            disconnectBtn: document.getElementById('disconnectBtn'),
            sendBtn: document.getElementById('sendBtn'),
            sendBroadcastBtn: document.getElementById('sendBroadcastBtn'),
            sendPrivateBtn: document.getElementById('sendPrivateBtn'),
            autoConnectBtn: document.getElementById('autoConnectBtn'),
            stressTestBtn: document.getElementById('stressTestBtn'),
            clearBtn: document.getElementById('clearBtn'),
            pingTime: document.getElementById('pingTime'),
            roomIndicator: document.getElementById('roomIndicator'),
            userCount: document.getElementById('userCount'),
            messageArea: document.getElementById('messageArea'),
            messageInput: document.getElementById('messageInput'),
            joinRoomBtn: document.getElementById('joinRoomBtn'),
            leaveRoomBtn: document.getElementById('leaveRoomBtn'),
            getRoomInfoBtn: document.getElementById('getRoomInfoBtn'),
            pingBtn: document.getElementById('pingBtn'),
            getStatusBtn: document.getElementById('getStatusBtn'),
            clearBtn: document.getElementById('clearBtn'),
            sendPrivateBtn: document.getElementById('sendPrivateBtn'),
            autoConnectBtn: document.getElementById('autoConnectBtn'),
            stressTestBtn: document.getElementById('stressTestBtn') ,
             connectionStatus: document.getElementById('connectionStatus')
        };
        this.elements.serverUrl.value = `ws://${window.location.host}/ws`;
         
        this.bindEvents();
        this.updateUI();
        
        // 设置初始用户名
        this.elements.username.value = this.username;
    }
    
    bindEvents() {
        // 连接按钮
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        
        // 消息发送
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        this.elements.sendBroadcastBtn.addEventListener('click', () => this.sendBroadcast());
        this.elements.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // 房间控制
        this.elements.joinRoomBtn.addEventListener('click', () => this.joinRoom());
        this.elements.leaveRoomBtn.addEventListener('click', () => this.leaveRoom());
        this.elements.getRoomInfoBtn.addEventListener('click', () => this.getRoomInfo());
        
        // 系统命令
        this.elements.pingBtn.addEventListener('click', () => this.sendPing());
        this.elements.getStatusBtn.addEventListener('click', () => this.getServerStatus());
        this.elements.clearBtn.addEventListener('click', () => this.clearMessages());
        
        // 私聊
        this.elements.sendPrivateBtn.addEventListener('click', () => this.sendPrivateMessage());
        
        // 高级功能
        this.elements.autoConnectBtn.addEventListener('click', () => this.testAutoReconnect());
        this.elements.stressTestBtn.addEventListener('click', () => this.stressTest());
        
        // 输入框变化
        this.elements.username.addEventListener('change', (e) => {
            this.username = e.target.value || this.username;
        });
        
        this.elements.roomId.addEventListener('change', (e) => {
            this.roomId = e.target.value || 'room1';
        });
    }
    
    connect() {
        if (this.isConnected) return;
        
        const serverUrl = this.elements.serverUrl.value;
        if (!serverUrl) {
            this.logError('请输入服务器地址');
            return;
        }
        
        this.updateStatus('连接中...', 'connecting');
        
        try {
            this.ws = new WebSocket(serverUrl);
            
            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onerror = (error) => this.onError(error);
            this.ws.onclose = () => this.onClose();
            
        } catch (error) {
            this.logError('创建WebSocket连接失败: ' + error.message);
            this.updateStatus('连接失败', 'disconnected');
        }
    }
    
    onOpen() {
        this.isConnected = true;
        this.connectedAt = new Date();
        this.reconnectAttempts = 0;
        this.stats.reconnects++;
        
        this.updateStatus('已连接', 'connected');
        this.log('WebSocket连接已建立');
        
        // 更新UI状态
        this.updateUI();
        
        // 开始心跳检测
        this.startPing();
        
        // 自动加入默认房间
        setTimeout(() => this.joinRoom(), 500);
    }
    
    onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
            this.stats.messagesReceived++;
            
        } catch (error) {
            // 如果不是JSON，可能是普通文本
            this.addMessage({
                type: 'echo',
                content: event.data,
                timestamp: new Date().toISOString()
            });
        }
        
        this.updateStats();
    }
    
    handleMessage(data) {
        const messageType = data.type;
        const timestamp = new Date(data.timestamp || Date.now()).toLocaleTimeString();
        
        switch (messageType) {
            case 'welcome':
                this.clientId = data.client_id;
                this.addMessage({
                    type: 'system',
                    username: '系统',
                    content: data.message,
                    timestamp: timestamp,
                    clientId: data.client_id
                });
                this.log('收到欢迎消息，客户端ID: ' + data.client_id);
                break;
                
            case 'pong':
                if (this.lastPingTime) {
                    this.pingLatency = Date.now() - this.lastPingTime;
                    this.elements.pingTime.textContent = this.pingLatency + 'ms';
                }
                this.log('收到Pong响应，延迟: ' + this.pingLatency + 'ms');
                break;
                
            case 'room_joined':
                this.addMessage({
                    type: 'system',
                    username: '系统',
                    content: `已加入房间: ${data.room_id} (${data.room_users}人在线)`,
                    timestamp: timestamp
                });
                this.elements.roomIndicator.innerHTML = 
                    `<span class="room-badge">${data.room_id}</span>`;
                this.elements.userCount.textContent = data.room_users + ' 用户';
                this.log(`加入房间成功: ${data.room_id}`);
                break;
                
            case 'user_joined':
                this.addMessage({
                    type: 'system',
                    username: '系统',
                    content: `${data.username} 加入了房间`,
                    timestamp: timestamp
                });
                this.updateUserList();
                break;
                
            case 'user_left':
                this.addMessage({
                    type: 'system',
                    username: '系统',
                    content: `${data.username} 离开了房间`,
                    timestamp: timestamp
                });
                this.updateUserList();
                break;
                
            case 'chat_message':
                this.addMessage({
                    type: 'chat',
                    username: data.username,
                    content: data.message,
                    timestamp: timestamp,
                    isSelf: data.client_id === this.clientId
                });
                break;
                
            case 'broadcast':
                this.addMessage({
                    type: 'broadcast',
                    username: data.username,
                    content: data.message,
                    timestamp: timestamp
                });
                break;
                
            case 'private_message':
                this.addMessage({
                    type: 'private',
                    username: data.from_username + ' (私聊)',
                    content: data.message,
                    timestamp: timestamp
                });
                // 播放提示音
                this.playNotificationSound();
                break;
                
            case 'private_message_sent':
                this.log('私聊消息已发送给: ' + data.to_client_id);
                break;
                
            case 'room_info':
                this.displayRoomInfo(data);
                break;
                
            case 'client_connected':
                this.log('新客户端连接: ' + data.client_id);
                break;
                
            case 'admin_message':
                this.addMessage({
                    type: 'admin',
                    username: '管理员',
                    content: data.message,
                    timestamp: timestamp
                });
                break;
                
            case 'error':
                this.addMessage({
                    type: 'error',
                    username: '错误',
                    content: data.message,
                    timestamp: timestamp
                });
                break;
                
            case 'echo':
                this.addMessage({
                    type: 'echo',
                    username: '回显',
                    content: JSON.stringify(data.original_message),
                    timestamp: timestamp
                });
                break;
                
            default:
                this.log('收到未知消息类型: ' + messageType);
                console.log('原始数据:', data);
        }
    }
    
    onError(error) {
        this.logError('WebSocket错误: ' + (error.message || '未知错误'));
        this.stats.errors++;
        this.updateStats();
    }
    
    onClose() {
        this.isConnected = false;
        this.updateStatus('已断开', 'disconnected');
        this.log('WebSocket连接已关闭');
        
        // 停止心跳
        this.stopPing();
        
        // 更新UI
        this.updateUI();
        
        // 尝试重连
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
            
            this.log(`尝试重新连接 (${this.reconnectAttempts}/${this.maxReconnectAttempts})，${delay}ms后重试`);
            
            setTimeout(() => {
                if (!this.isConnected) {
                    this.connect();
                }
            }, delay);
        }
    }
    
    disconnect() {
        if (!this.ws) return;
        
        this.ws.close(1000, '用户手动断开');
        this.ws = null;
        this.isConnected = false;
        
        this.updateStatus('已断开', 'disconnected');
        this.updateUI();
        this.log('已主动断开连接');
    }
    
    sendMessage() {
        if (!this.isConnected || !this.ws) {
            this.logError('未连接，无法发送消息');
            return;
        }
        
        const messageInput = this.elements.messageInput;
        const message = messageInput.value.trim();
        
        if (!message) {
            this.logError('消息不能为空');
            return;
        }
        
        const messageData = {
            type: 'chat_message',
            message: message,
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(messageData));
        messageInput.value = '';
        
        this.stats.messagesSent++;
        this.updateStats();
        
        // 添加到本地消息列表
        this.addMessage({
            type: 'chat',
            username: this.username + ' (我)',
            content: message,
            timestamp: new Date().toLocaleTimeString(),
            isSelf: true
        });
    }
    
    sendBroadcast() {
        if (!this.isConnected || !this.ws) {
            this.logError('未连接，无法发送广播');
            return;
        }
        
        const message = prompt('请输入广播消息:');
        if (!message) return;
        
        const messageData = {
            type: 'broadcast',
            message: message,
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(messageData));
        this.log('广播消息已发送: ' + message);
    }
    
    sendPrivateMessage() {
        if (!this.isConnected || !this.ws) {
            this.logError('未连接，无法发送私聊');
            return;
        }
        
        const targetSelect = this.elements.privateUserSelect;
        const privateMessageInput = this.elements.privateMessage;
        
        const targetClientId = targetSelect.value;
        const message = privateMessageInput.value.trim();
        
        if (!targetClientId) {
            this.logError('请选择私聊对象');
            return;
        }
        
        if (!message) {
            this.logError('私聊消息不能为空');
            return;
        }
        
        const messageData = {
            type: 'private_message',
            target_client_id: targetClientId,
            message: message,
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(messageData));
        privateMessageInput.value = '';
        
        this.log(`私聊消息已发送给: ${targetSelect.options[targetSelect.selectedIndex].text}`);
    }
    
    sendPing() {
        if (!this.isConnected || !this.ws) {
            this.logError('未连接，无法发送Ping');
            return;
        }
        
        this.lastPingTime = Date.now();
        const pingData = {
            type: 'ping',
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(pingData));
        this.log('Ping已发送');
    }
    
    joinRoom() {
        if (!this.isConnected || !this.ws) {
            this.logError('未连接，无法加入房间');
            return;
        }
        
        const roomId = this.elements.roomId.value || 'room1';
        const username = this.elements.username.value || this.username;
        
        this.roomId = roomId;
        this.username = username;
        
        const joinData = {
            type: 'join_room',
            room_id: roomId,
            username: username,
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(joinData));
        this.log(`请求加入房间: ${roomId}, 用户名: ${username}`);
    }
    
    leaveRoom() {
        if (!this.isConnected || !this.ws) {
            this.logError('未连接，无法离开房间');
            return;
        }
        
        const leaveData = {
            type: 'leave_room',
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(leaveData));
        this.log('已请求离开房间');
        
        // 清空房间指示器
        this.elements.roomIndicator.innerHTML = '';
        this.elements.userCount.textContent = '0 用户';
    }
    
    getRoomInfo() {
        if (!this.isConnected || !this.ws) {
            this.logError('未连接，无法获取房间信息');
            return;
        }
        
        const roomData = {
            type: 'get_room_info',
            room_id: this.roomId,
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(roomData));
        this.log('请求获取房间信息');
    }
    
    getServerStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                const serverInfo = this.elements.serverInfo;
                serverInfo.innerHTML = `
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <p><strong>服务器状态:</strong> ${data.status}</p>
                        <p><strong>连接数:</strong> ${data.connected_clients}</p>
                        <p><strong>房间数:</strong> ${Object.keys(data.rooms || {}).length}</p>
                        <p><strong>时间:</strong> ${new Date(data.timestamp).toLocaleTimeString()}</p>
                    </div>
                `;
                this.log('获取服务器状态成功');
            })
            .catch(error => {
                this.logError('获取服务器状态失败: ' + error.message);
            });
    }
    
    addMessage(msg) {
        const messageArea = this.elements.messageArea;
        const messageDiv = document.createElement('div');
        
        messageDiv.className = 'message';
        
        if (msg.type === 'system') {
            messageDiv.className += ' message-system';
        } else if (msg.type === 'error') {
            messageDiv.className += ' message-error';
        } else if (msg.type === 'private') {
            messageDiv.className += ' message-private';
        } else if (msg.type === 'admin') {
            messageDiv.className += ' message-admin';
        }
        
        messageDiv.innerHTML = `
            <div>
                <span class="message-username">${msg.username || '未知用户'}</span>
                <span class="message-timestamp">${msg.timestamp}</span>
            </div>
            <div class="message-content">${this.escapeHtml(msg.content)}</div>
        `;
        
        messageArea.appendChild(messageDiv);
        messageArea.scrollTop = messageArea.scrollHeight;
        
        this.messageCount++; 
        document.getElementById('messageCount').textContent = this.messageCount;
    }
    
    clearMessages() {
        document.getElementById('messageArea').innerHTML = '';
        this.messageCount = 0;
        document.getElementById('messageCount').textContent = '0';
        this.log('已清空消息记录');
    }
    
    updateUserList() {
        // 这里可以扩展为从服务器获取用户列表
        // 目前只是占位符
        const userListArea = document.getElementById('userListArea');
        userListArea.innerHTML = '<p>用户列表需要服务器支持...</p>';
    }
    
    displayRoomInfo(data) {
        const messageArea = document.getElementById('messageArea');
        const infoDiv = document.createElement('div');
        
        infoDiv.className = 'message message-system';
        infoDiv.innerHTML = `
            <div>
                <span class="message-username">房间信息</span>
                <span class="message-timestamp">${new Date(data.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="message-content">
                <p><strong>房间ID:</strong> ${data.room_id}</p>
                <p><strong>在线用户:</strong> ${data.user_count}人</p>
                <p><strong>用户列表:</strong></p>
                <ul>
                    ${data.users ? data.users.map(user => 
                        `<li>${user.username} (${user.client_id.substring(0, 8)}...)</li>`
                    ).join('') : '<li>暂无用户</li>'}
                </ul>
            </div>
        `;
        
        messageArea.appendChild(infoDiv);
        messageArea.scrollTop = messageArea.scrollHeight;
        
        // 更新私聊用户选择
        this.updatePrivateUserSelect(data.users);
    }
    
    updatePrivateUserSelect(users) {
        const select = document.getElementById('privateUserSelect');
        select.innerHTML = '<option value="">选择用户...</option>';
        
        if (users && Array.isArray(users)) {
            users.forEach(user => {
                if (user.client_id !== this.clientId) {
                    const option = document.createElement('option');
                    option.value = user.client_id;
                    option.textContent = `${user.username} (${user.client_id.substring(0, 8)}...)`;
                    select.appendChild(option);
                }
            });
            
            select.disabled = false;
            document.getElementById('privateMessage').disabled = false;
            document.getElementById('sendPrivateBtn').disabled = false;
        }
    }
    
    startPing() {
        // 每10秒发送一次ping
        this.pingInterval = setInterval(() => {
            if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.sendPing();
            }
        }, 10000);
    }
    
    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    updateStatus(text, status) {
        const statusDisplay = document.getElementById('statusDisplay');
        const connectionStatus = this.elements.connectionStatus;
        
        statusDisplay.textContent = text;
        statusDisplay.className = 'status status-' + status;
        
        connectionStatus.textContent = text;
        connectionStatus.style.color = status === 'connected' ? '#4CAF50' : 
                                       status === 'connecting' ? '#FF9800' : '#F44336';
    }
    
    updateUI() {
        const isConnected = this.isConnected;
        
        // 按钮状态
        this.elements.connectBtn.disabled = isConnected;
        this.elements.disconnectBtn.disabled = !isConnected;
        this.elements.sendBtn.disabled = !isConnected;
        this.elements.sendBroadcastBtn.disabled = !isConnected;
        this.elements.pingBtn.disabled = !isConnected;
        this.elements.joinRoomBtn.disabled = !isConnected;
        this.elements.leaveRoomBtn.disabled = !isConnected;
        this.elements.getRoomInfoBtn.disabled = !isConnected;
        
        // 连接状态指示器
        const dot = document.createElement('span');
        dot.className = 'connection-dot dot-' + (isConnected ? 'connected' : 'disconnected');
        document.getElementById('connectionStatus').prepend(dot);
    }
    
    updateStats() {
        // 更新连接时长
        if (this.connectedAt) {
            const connectedSeconds = Math.floor((new Date() - this.connectedAt) / 1000);
            document.getElementById('connectedTime').textContent = connectedSeconds + 's';
        }
        
        // 更新统计显示
        const statsDiv = document.createElement('div');
        statsDiv.style.marginTop = '10px';
        statsDiv.style.fontSize = '12px';
        statsDiv.style.color = '#666';
        statsDiv.innerHTML = `
            <div>发送: ${this.stats.messagesSent} | 接收: ${this.stats.messagesReceived} | 错误: ${this.stats.errors} | 重连: ${this.stats.reconnects}</div>
        `;
        
        // 更新或创建统计显示
        let existingStats = document.getElementById('clientStats');
        if (existingStats) {
            existingStats.innerHTML = statsDiv.innerHTML;
        } else {
            statsDiv.id = 'clientStats';
            document.getElementById('connectionLog').parentElement.appendChild(statsDiv);
        }
    }
    
    log(text) {
        const logArea = document.getElementById('connectionLog');
        const logEntry = document.createElement('div');
        logEntry.className = 'log';
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
        
        logArea.appendChild(logEntry);
        logArea.scrollTop = logArea.scrollHeight;
    }
    
    logError(text) {
        this.log(`❌ ${text}`);
        console.error(text);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    playNotificationSound() {
        // 简单的通知声音
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.1);
        } catch (error) {
            // 音频API可能不可用，忽略
        }
    }
    
    // 测试功能
    testAutoReconnect() {
        this.log('开始自动重连测试...');
        this.disconnect();
        
        setTimeout(() => {
            this.connect();
        }, 1000);
    }
    
    stressTest() {
        this.log('开始压力测试 (发送10条测试消息)...');
        
        for (let i = 1; i <= 10; i++) {
            setTimeout(() => {
                if (this.isConnected && this.ws) {
                    const testData = {
                        type: 'chat_message',
                        message: `压力测试消息 ${i}`,
                        timestamp: new Date().toISOString()
                    };
                    this.ws.send(JSON.stringify(testData));
                    this.log(`发送压力测试消息 ${i}`);
                }
            }, i * 100);
        }
    }
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', () => {
    window.wsClient = new WebSocketClient();
    
    // 可选：自动连接
    // setTimeout(() => window.wsClient.connect(), 1000);
    
    // 添加键盘快捷键
    document.addEventListener('keydown', (e) => {
        // Ctrl+Enter 发送消息
        if (e.ctrlKey && e.key === 'Enter') {
            window.wsClient.sendMessage();
        }
        // Ctrl+P 发送ping
        if (e.ctrlKey && e.key === 'p') {
            e.preventDefault();
            window.wsClient.sendPing();
        }
        // Ctrl+R 重新连接
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            window.wsClient.connect();
        }
        // Ctrl+D 断开连接
        if (e.ctrlKey && e.key === 'd') {
            e.preventDefault();
            window.wsClient.disconnect();
        }
    });
    
    console.log('WebSocket客户端已初始化，使用 window.wsClient 访问');
});