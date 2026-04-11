import { FFmpeg } from '/static/assets/ffmpeg/package/dist/esm/index.js';
import { fetchFile } from '/static/assets/util/package/dist/esm/index.js';

class RTSPPlayer {
	constructor() {
		this.ffmpeg = null;
		this.socket = null;
		this.isPlaying = false;
		this.frameCount = 0;
		this.startTime = null;
		this.chunkBuffer = [];

		// DOM元素
		this.canvas = document.getElementById('videoCanvas');
		this.ctx = this.canvas.getContext('2d');
		this.statusEl = document.getElementById('status');
		this.statsEl = document.getElementById('stats');
		this.wsUrlEl = document.getElementById('wsUrl');
		this.wsUrlEl.value = `ws://${window.location.host}/ws/stream`;

		// 事件绑定
		this.bindEvents();
	}

	bindEvents() {
		document
			.getElementById('connectBtn')
			.addEventListener('click', () => this.connect());
		document
			.getElementById('disconnectBtn')
			.addEventListener('click', () => this.disconnect());
		document.getElementById('testBtn1').addEventListener('click', () => {
			document.getElementById('rtspUrl').value =
				'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4';
		});
		document.getElementById('testBtn2').addEventListener('click', () => {
			document.getElementById('rtspUrl').value =
				'rtsp://admin:123456@192.168.1.100:554/stream1';
		});
		document.getElementById('sendBtn').addEventListener('click', () => {
			this.sendMessage();
		});
	}

	async initFFmpeg() {
		if (this.ffmpeg && this.ffmpeg.loaded) {
			return true;
		}

		this.setStatus('正在初始化ffmpeg.wasm...');

		try {
			this.ffmpeg = new FFmpeg();

			// 事件监听
			this.ffmpeg.on('log', ({ message }) => {
				console.log('[FFmpeg Log]', message);
			});

			this.ffmpeg.on('progress', ({ progress }) => {
				this.setStats(`解码进度: ${(progress * 100).toFixed(1)}%`);
			});

			// 加载核心
			await this.ffmpeg.load({
				coreURL: '/static/assets/core/package/dist/esm/ffmpeg-core.js',
			});

			this.setStatus('ffmpeg.wasm初始化完成');
			return true;
		} catch (error) {
			// ReferenceError: SharedArrayBuffer is not defined
			this.setStatus(`ffmpeg初始化失败: ${error.message}`);
			console.error('FFmpeg初始化错误:', error);
			return false;
		}
	}

	async connect() {
		if (this.isPlaying) {
			return;
		}
		const rtspUrl = document.getElementById('rtspUrl').value.trim();
		if (!rtspUrl) {
			alert('请输入RTSP URL');
			return;
		}

		// 初始化ffmpeg
		if (!(await this.initFFmpeg())) {
			return;
		}

		this.setStatus('正在连接WebSocket...');

		try {
			// 构建WebSocket URL
			//const wsUrl = `ws://${window.location.host}/ws/stream?url=${encodeURIComponent(rtspUrl)}`;
			const streamUrl = encodeURIComponent(rtspUrl);
			const wsUrl = `${this.wsUrlEl.value}?url=${streamUrl}`;
			this.socket = new WebSocket(wsUrl);
			this.socket.binaryType = 'arraybuffer';

			this.socket.onopen = () => {
				this.isPlaying = true;
				this.startTime = Date.now();
				this.frameCount = 0;
				this.setStatus('已连接，正在接收流数据...');
				this.updateButtons();
				this.startRendering();
			};

			this.socket.onmessage = async (event) => {
				if (!this.isPlaying) return;

				const chunk = new Uint8Array(event.data);
				this.chunkBuffer.push(chunk);

				// 统计信息
				this.frameCount++;
				const elapsed = (Date.now() - this.startTime) / 1000;
				const fps = (this.frameCount / elapsed).toFixed(1);
				this.setStats(
					`FPS: ${fps} | 帧数: ${this.frameCount} | 数据块: ${this.chunkBuffer.length}`,
				);

				// 当积累足够数据时解码
				if (this.chunkBuffer.length >= 3) {
					await this.decodeChunks();
				}
			};

			this.socket.onerror = (error) => {
				console.error('WebSocket错误:', error);
				this.setStatus('连接错误');
				this.disconnect();
			};

			this.socket.onclose = () => {
				console.log('WebSocket连接关闭');
				this.disconnect();
			};
		} catch (error) {
			this.setStatus(`连接失败: ${error.message}`);
			console.error('连接错误:', error);
		}
	}

	async decodeChunks() {
		if (!this.isPlaying || this.chunkBuffer.length === 0) {
			return;
		}

		// 合并数据块
		const totalSize = this.chunkBuffer.reduce(
			(sum, chunk) => sum + chunk.length,
			0,
		);
		const mergedData = new Uint8Array(totalSize);
		let offset = 0;

		for (const chunk of this.chunkBuffer) {
			mergedData.set(chunk, offset);
			offset += chunk.length;
		}
		//console.info(mergedData)
		this.chunkBuffer = []; // 清空缓冲区
		try {
			/*
			// 写入文件
			await this.ffmpeg.writeFile('input.mp4', mergedData);

			// 使用ffmpeg解码
			// 爆内存
			await this.ffmpeg.exec([
				'-i','input.mp4',
				'-vf','fps=15,scale=640:480',
				'-f','rawvideo',
				'-pix_fmt','rgba',
				'output.rgba',
			]);
			 

			// 读取解码后的数据
			const rgbaData = await this.ffmpeg.readFile('output.rgba');

			// 渲染到canvas
			this.renderToCanvas(new Uint8ClampedArray(rgbaData.buffer));

			// 清理临时文件
			await this.ffmpeg.deleteFile('input.mp4');
			await this.ffmpeg.deleteFile('output.rgba');

*/


			// 清理旧文件，防止内存残留
			//await this.ffmpeg.FS('unlink', 'input.mp4').catch(() => {});
			//await this.ffmpeg.FS('unlink', 'frame_%04d.png').catch(() => {});
			await this.ffmpeg.deleteFile('input.mp4');
			await this.ffmpeg.deleteFile('output.rgba');

			// 写入输入视频
			await this.ffmpeg.writeFile('input.mp4', this.mergedData);

			// ✅ 安全命令：输出 PNG 序列
			await this.ffmpeg.exec([
				'-i', 'input.mp4',
				'-vf', 'fps=15,scale=640:480',
				'-f', 'image2',
				'-pix_fmt', 'rgba',
				'frame_%04d.png'
			]);

			// 读取第一帧示例
			await this.ffmpeg.readFile('frame_0001.png');
		} catch (error) {
			console.warn('解码错误:', error);
			// 如果解码失败，跳过这些数据继续
		}
	}

	renderToCanvas(rgbaData) {
		if (rgbaData.length !== 640 * 480 * 4) {
			console.warn('数据尺寸不匹配');
			return;
		}

		const imageData = new ImageData(rgbaData, 640, 480);
		this.ctx.putImageData(imageData, 0, 0);
	}

	startRendering() {
		const renderLoop = () => {
			if (!this.isPlaying) return;

			// 这里可以添加其他渲染逻辑
			requestAnimationFrame(renderLoop);
		};

		renderLoop();
	}
	sendMessage() {
		const message = document.getElementById('message').value.trim();
		if (!message) {
			alert('请输入要发送的消息');
			return;
		}
		this.socket.send(message);
		this.updateButtons();
	}
	disconnect() {
		if (!this.isPlaying) return;

		this.isPlaying = false;
		this.setStatus('正在断开连接...');

		if (this.socket && this.socket.readyState === WebSocket.OPEN) {
			this.socket.send('close');
			this.socket.close();
		}

		this.socket = null;
		this.chunkBuffer = [];
		this.updateButtons();

		// 清理canvas
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.setStatus('已断开连接');
		this.setStats('');
	}

	setStatus(message) {
		this.statusEl.textContent = `状态: ${message}`;
	}

	setStats(info) {
		this.statsEl.textContent = info;
	}

	updateButtons() {
		const connectBtn = document.getElementById('connectBtn');
		const disconnectBtn = document.getElementById('disconnectBtn');
		const sendBtn = document.getElementById('sendBtn');

		connectBtn.disabled = this.isPlaying;
		disconnectBtn.disabled = !this.isPlaying;
		sendBtn.disabled = !this.isPlaying;
	}
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', () => {
	const player = new RTSPPlayer();
	window.player = player; // 调试用

	// 页面关闭前断开连接
	window.addEventListener('beforeunload', () => {
		player.disconnect();
	});

	console.log('RTSP播放器已初始化');
});
