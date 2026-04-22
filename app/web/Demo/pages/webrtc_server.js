// signaling-server.js
const WebSocket = require('ws');
const http = require('http');

class SignalingServer {
    constructor(port = 8765) {
        this.port = port;
        this.clients = new Map(); // peerId -> { ws, roomId }
        this.rooms = new Map();   // roomId -> Set of peerIds
        
        this.server = http.createServer();
        this.wss = new WebSocket.Server({ server: this.server });
        
        this.setupWebSocket();
        this.start();
    }
    
    setupWebSocket() {
        this.wss.on('connection', (ws) => {
            console.log('新的客户端连接');
            
            ws.on('message', (message) => {
                try {
                    const data = JSON.parse(message);
                    this.handleMessage(ws, data);
                } catch (error) {
                    console.error('消息解析错误:', error);
                }
            });
            
            ws.on('close', () => {
                this.handleDisconnect(ws);
            });
            
            ws.on('error', (error) => {
                console.error('WebSocket 错误:', error);
            });
        });
    }
    
    handleMessage(ws, data) {
        switch (data.type) {
            case 'register':
                this.handleRegister(ws, data);
                break;
            case 'offer':
                this.handleOffer(data);
                break;
            case 'answer':
                this.handleAnswer(data);
                break;
            case 'ice-candidate':
                this.handleIceCandidate(data);
                break;
            case 'call-request':
                this.handleCallRequest(data);
                break;
            case 'call-accepted':
                this.handleCallAccepted(data);
                break;
            case 'call-rejected':
                this.handleCallRejected(data);
                break;
            case 'hangup':
                this.handleHangup(data);
                break;
            default:
                console.log('未知消息类型:', data.type);
        }
    }
    
    handleRegister(ws, data) {
        const { peerId, roomId } = data;
        
        // 检查 peerId 是否已存在
        if (this.clients.has(peerId)) {
            this.sendTo(ws, {
                type: 'error',
                message: 'Peer ID 已存在'
            });
            return;
        }
        
        // 注册客户端
        this.clients.set(peerId, { ws, roomId });
        
        // 加入房间
        if (!this.rooms.has(roomId)) {
            this.rooms.set(roomId, new Set());
        }
        this.rooms.get(roomId).add(peerId);
        
        // 发送欢迎消息
        this.sendTo(ws, {
            type: 'welcome',
            peerId,
            roomId
        });
        
        // 发送房间中的用户列表
        this.sendPeerList(ws, roomId);
        
        // 通知房间中的其他用户
        this.broadcastToRoom(roomId, peerId, {
            type: 'peer-joined',
            peerId
        });
        
        console.log(`客户端注册: ${peerId}, 房间: ${roomId}`);
    }
    
    handleOffer(data) {
        const { to, from, sdp } = data;
        this.forwardToPeer(to, {
            type: 'offer',
            sdp,
            from
        });
    }
    
    handleAnswer(data) {
        const { to, from, sdp } = data;
        this.forwardToPeer(to, {
            type: 'answer',
            sdp,
            from
        });
    }
    
    handleIceCandidate(data) {
        const { to, from, candidate } = data;
        this.forwardToPeer(to, {
            type: 'ice-candidate',
            candidate,
            from
        });
    }
    
    handleCallRequest(data) {
        const { to, from } = data;
        this.forwardToPeer(to, {
            type: 'call-request',
            from
        });
    }
    
    handleCallAccepted(data) {
        const { to, from } = data;
        this.forwardToPeer(to, {
            type: 'call-accepted',
            from
        });
    }
    
    handleCallRejected(data) {
        const { to, from } = data;
        this.forwardToPeer(to, {
            type: 'call-rejected',
            from
        });
    }
    
    handleHangup(data) {
        const { to, from } = data;
        this.forwardToPeer(to, {
            type: 'hangup',
            from
        });
    }
    
    handleDisconnect(ws) {
        // 找到断开的客户端
        let disconnectedPeerId = null;
        let disconnectedRoomId = null;
        
        for (const [peerId, client] of this.clients.entries()) {
            if (client.ws === ws) {
                disconnectedPeerId = peerId;
                disconnectedRoomId = client.roomId;
                break;
            }
        }
        
        if (disconnectedPeerId) {
            // 从客户端列表移除
            this.clients.delete(disconnectedPeerId);
            
            // 从房间移除
            if (disconnectedRoomId && this.rooms.has(disconnectedRoomId)) {
                const room = this.rooms.get(disconnectedRoomId);
                room.delete(disconnectedPeerId);
                
                // 如果房间为空，删除房间
                if (room.size === 0) {
                    this.rooms.delete(disconnectedRoomId);
                } else {
                    // 通知房间中的其他用户
                    this.broadcastToRoom(disconnectedRoomId, disconnectedPeerId, {
                        type: 'peer-left',
                        peerId: disconnectedPeerId
                    });
                }
            }
            
            console.log(`客户端断开: ${disconnectedPeerId}`);
        }
    }
    
    forwardToPeer(peerId, message) {
        const client = this.clients.get(peerId);
        if (client && client.ws.readyState === WebSocket.OPEN) {
            this.sendTo(client.ws, message);
        }
    }
    
    sendTo(ws, message) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
        }
    }
    
    sendPeerList(ws, roomId) {
        if (this.rooms.has(roomId)) {
            const peers = Array.from(this.rooms.get(roomId));
            this.sendTo(ws, {
                type: 'peer-list',
                peers
            });
        }
    }
    
    broadcastToRoom(roomId, excludePeerId, message) {
        if (this.rooms.has(roomId)) {
            const room = this.rooms.get(roomId);
            for (const peerId of room) {
                if (peerId !== excludePeerId) {
                    this.forwardToPeer(peerId, message);
                }
            }
        }
    }
    
    start() {
        this.server.listen(this.port, () => {
            console.log(`信令服务器运行在 http://localhost:${this.port}`);
            console.log(`WebSocket 端点: ws://localhost:${this.port}`);
        });
    }
}

// 启动服务器
const server = new SignalingServer(8765);