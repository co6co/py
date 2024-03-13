import { type talkerMessageData } from './src/types';
class xTalker {
	xtalk_xss_mode: boolean = false;
	xtalk_constraints: any = { video: false, audio: true };
	xtalk_configuration: RTCConfiguration | undefined;
	xtalk_want_remote_offerer = false;
	xtalk_audio_element_id = 'audiostrm';
	xtalk_device_addr: string | undefined;
	xtalk_device_port: number | undefined;
	// Set this to override the automatic detection in xtalk_websocket_server_connect()
	xtalk_xss_to_device_id = -1;
	xtalk_xss_to_gb_url = 'none';
	xtalk_xss_addr: string | undefined;
	xtalk_xss_port: number | undefined;
	xtalk_conn_state = 'Disconnected';
	xTalkWebrtcPeerConnection: any;
	xTalkWebsocketConnection: any;
	xTalkConnectAttempts = 0;
	xTalkPeerID: string | undefined;
	xTalkCallID: number = -1;
	xTalkSessionID: string | undefined;
	// Promise for local stream after constraints are approved by the user
	xTalkLocalStrmPromise: any;

	xTalkSetConnectState(value: string) {
		this.xtalk_conn_state = value;
	}

	xTalkResetState() {
		// This will call xTalkOnServerClose()
		this.xtalk_audio_element_id = 'audiostrm';
		this.xtalk_xss_to_gb_url = 'none';
		this.xTalkWebsocketConnection.close();
	}

	xtalk_websocket_server_disconn() {
		this.xTalkResetState();
	}

	xTalkHandleIncomingError(error: any) {
		this.xTalkSetError('ERROR: ' + error);
		this.xTalkResetState();
	}

	xTalkGetAudioElement(): HTMLMediaElement {
		let ele = document.getElementById(this.xtalk_audio_element_id);
		if (ele && 'srcObject' in ele) return ele as HTMLMediaElement;
		this.xTalkSetError('ERROR: ' + HTMLMediaElement);
		throw new Error(`id:${this.xtalk_audio_element_id} ,没有 srcObject 属性`);
	}

	xTalkSetStatus(text: string) {
		console.log(text);
		var span = document.getElementById('status');
		// Don't set the status if it already contains an error
		if (!span) return;
		if (!span.classList.contains('error')) span.textContent = text;
	}

	xTalkSetError(text: string) {
		console.error(text);
		var span = document.getElementById('status');
		if (!span) return;
		span.textContent = text;
		span.classList.add('error');
	}

	// SDP offer received from peer, set remote description and create an answer
	xTalkoOnIncomingSDP(sdp: any) {
		let that = this;
		this.xTalkWebrtcPeerConnection
			.setRemoteDescription(sdp)
			.then(() => {
				that.xTalkSetStatus('Remote SDP set');
				if (sdp.type != 'offer') return;
				that.xTalkSetStatus('Got SDP offer');
				that.xTalkLocalStrmPromise
					.then((stream: any) => {
						that.xTalkSetStatus('Got local stream, creating answer');
						that.xTalkWebrtcPeerConnection
							.createAnswer()
							.then(that.xTalkOnLocalDescription.bind(that))
							.catch(that.xTalkSetError);
					})
					.catch(that.xTalkSetError);
			})
			.catch(that.xTalkSetError);
	}

	// Local description was set, send it to peer
	xTalkOnLocalDescription(desc: any) {
		let that = this;
		console.log('Got local description: ' + JSON.stringify(desc));
		this.xTalkWebrtcPeerConnection.setLocalDescription(desc).then(function () {
			that.xTalkSetStatus('Sending SDP ' + desc.type);
			let sdp = { sdp: that.xTalkWebrtcPeerConnection.localDescription };

			if (!that.xtalk_xss_mode)
				that.xTalkWebsocketConnection.send(JSON.stringify(sdp));
			else
				that.xTalkWebsocketConnection.send(
					'PEER_MSG session_id=' +
						that.xTalkSessionID +
						' ' +
						JSON.stringify(sdp)
				);
		});
	}

	xTalkOnGenerateOffer() {
		this.xTalkWebrtcPeerConnection
			.createOffer()
			.then(this.xTalkOnLocalDescription)
			.catch(this.xTalkSetError);
	}

	// ICE candidate received from peer, add it to the peer connection
	xTalkOnIncomingICE(ice: any) {
		var candidate = new RTCIceCandidate(ice);
		this.xTalkWebrtcPeerConnection
			.addIceCandidate(candidate)
			.catch(this.xTalkSetError);
	}
	//这里的做法就是用到那个字段去解析那个字段
	onMessage(data:talkerMessageData){ 
		console.log('Received ' + data); 
	}
	xTalkOnServerMessageXss(event: any) {
		var msg;
		var msg_splits;
		var sdp_ice_msg; 
		msg_splits = event.data.split(' ');
		//['PEER_MSG', 'session_id=4446fbdf-9be0-4900-818d-b0a6138eca3f', '{"ice":{"sdpMLineIndex":0,"candidate":"candidate:4', '1', 'UDP', '2015363583', '10.6.3.135', '33009', 'typ', 'host"}}']
		
		if (msg_splits.length == 2 && msg_splits[0] == 'REGIST_OK') {
			//REGIST_OK ****
			let peer_id = msg_splits[1];

			console.log('Regist to ccws successed: ' + peer_id);

			this.xTalkCallID = 1; 
			var to_peer_str;
			if (this.xtalk_xss_to_gb_url != 'none')
				to_peer_str = 'device=-1 track=-1 gb=' + this.xtalk_xss_to_gb_url;
			else
				to_peer_str =
					'device=' + this.xtalk_xss_to_device_id + ' track=-1 gb=none';

			this.xTalkWebsocketConnection.send(
				'SESSION call_id=' + this.xTalkCallID + ' media=talk ' + to_peer_str
			);
		} else if (msg_splits.length == 3 && msg_splits[0] == 'SESSION_OK') {
			//SESSION_OK call_id=* session_id=*
			var callid_splits, ssid_splits;
			var to_callid;

			callid_splits = msg_splits[1].split('=', 2);
			if (callid_splits.length == 2 && callid_splits[0] == 'call_id')
				to_callid = callid_splits[1];

			ssid_splits = msg_splits[2].split('=', 2);
			if (ssid_splits.length == 2 && ssid_splits[0] == 'session_id')
				this.xTalkSessionID = ssid_splits[1];

			console.log(
				'session established: ' +
					this.xTalkSessionID +
					' for call ' +
					this.xTalkCallID
			);
		} else if (msg_splits.length == 3 && msg_splits[0] == 'SESSION_REJ') {
			//SESSION_REJ call_id=* reason=*
			var callid_splits;
			var to_callid;
			var str_reason;

			callid_splits = msg_splits[1].split('=', 2);
			if (callid_splits.length == 2 && callid_splits[0] == 'call_id')
				to_callid = callid_splits[1];

			if (to_callid == null) return;

			reason_splits = msg_splits[2].split('=', 2);
			if (reason_splits.length == 2 && reason_splits[0] == 'reason')
				str_reason = reason_splits[1];

			//此处要释放跟call相关的资源
			console.log('call is rejected: ' + to_callid + ' reason:' + str_reason);
		} else if (msg_splits.length == 3 && msg_splits[0] == 'SESSION_BYE') {
			//SESSION_BYE session_id=* reason=*
			var ssid_splits;
			var reason_splits;
			var to_ssid;
			var str_reason;

			ssid_splits = msg_splits[1].split('=', 2);
			if (ssid_splits.length == 2 && ssid_splits[0] == 'session_id')
				to_ssid = ssid_splits[1];

			if (this.xTalkSessionID == null || to_ssid != this.xTalkSessionID) return;

			reason_splits = msg_splits[2].split('=', 2);
			if (reason_splits.length == 2 && reason_splits[0] == 'reason')
				str_reason = reason_splits[1];

			//此处要释放跟session相关的资源
			console.log(
				'session bye: ' + this.xTalkSessionID + ' reason:' + str_reason
			);
		} else if (msg_splits.length >= 2 && msg_splits[0] == 'PEER_MSG') {
			//PEER_MSG session_id=* sdp或ice消息串
			var ssid_splits;
			var to_ssid;

			ssid_splits = msg_splits[1].split('=', 2);
			if (ssid_splits.length == 2 && ssid_splits[0] == 'session_id'){
				to_ssid = ssid_splits[1];
				this.onMessage({SessionId:to_ssid})
			} 

			if (this.xTalkSessionID == null || to_ssid != this.xTalkSessionID) return;

			//找到第二个空格后的字符串，那是实际的sdp或ice消息
			var pos = 'PEER_MSG session_id='.length;
			var n = event.data.indexOf(' ', pos);
			sdp_ice_msg = event.data.substring(n + 1);

			try {
				msg = JSON.parse(sdp_ice_msg);
			} catch (e) {
				return;
			}

			// Incoming JSON signals the beginning of a call
			if (!this.xTalkWebrtcPeerConnection) this.xTalkCreateCall(msg);

			if (msg.sdp != null) {
				this.xTalkoOnIncomingSDP(msg.sdp);
			} else if (msg.ice != null) {
				this.xTalkOnIncomingICE(msg.ice);
			} else {
				this.xTalkHandleIncomingError('Unknown incoming JSON: ' + msg);
			}
		} else {
			console.log('unexpected message: ' + event.data);
		}
	}

	xTalkOnServerMessage(event: any) {  
		if (event.data.startsWith('ERROR')) {
			this.xTalkHandleIncomingError(event.data);
			return;
		}
		// Handle incoming JSON SDP and ICE messages
		let msg = null;
		try {
			msg = JSON.parse(event.data);
		} catch (e) {
			if (e instanceof SyntaxError) {
				this.xTalkHandleIncomingError(
					'Error parsing incoming JSON: ' + event.data
				);
			} else {
				this.xTalkHandleIncomingError(
					'Unknown error parsing response: ' + event.data
				);
			}
			return;
		}

		// Incoming JSON signals the beginning of a call
		if (!this.xTalkWebrtcPeerConnection) this.xTalkCreateCall(msg);

		if (msg.sdp != null) {
			this.xTalkoOnIncomingSDP(msg.sdp);
		} else if (msg.ice != null) {
			this.xTalkOnIncomingICE(msg.ice);
		} else {
			this.xTalkHandleIncomingError('Unknown incoming JSON: ' + msg);
		}
	}

	xTalkOnServerClose(event: any) {
		this.xTalkSetStatus('Disconnected from server');
		//resetVideo();

		if (this.xTalkWebrtcPeerConnection) {
			this.xTalkWebrtcPeerConnection.close();
			this.xTalkWebrtcPeerConnection = null;
		}

		this.xTalkSetConnectState('Disonnected');

		// Reset after a second
		//window.setTimeout(xtalk_websocket_server_connect, 1000);
	}

	xTalkOnServerError(event: any) {
		console.error(
			'Unable to connect to server, did you add an exception for the certificate?'
		);

		// Retry after 3 seconds
		//window.setTimeout(xtalk_websocket_server_connect, 3000);
	}

	xTalkGetLocalStream() {
		console.log(JSON.stringify(this.xtalk_constraints));
		// Add local stream
		if (navigator.mediaDevices.getUserMedia) {
			return navigator.mediaDevices.getUserMedia(this.xtalk_constraints);
		} else {
			this.xTalkErrorUserMediaHandler();
			throw new Error('not implemented');
		}
	}

	xtalk_websocket_server_connect() {
		let that =this
		//xTalkConnectAttempts++;

		//if (xTalkConnectAttempts > 3) {
		//    xTalkSetError("Too many connection attempts, aborting. Refresh page to try again");
		//    return;
		//}
		var default_hostname = '127.0.0.1';
		if (window.location.protocol.startsWith('http')) {
			default_hostname = window.location.hostname;
		}

		var ws_url;
		var ws_host;
		var ws_port;
		if (this.xtalk_xss_mode) {
			ws_host = this.xtalk_xss_addr || default_hostname;
			ws_port = this.xtalk_xss_port || '8440';
			ws_url =
				'wss://' +
				'stream.jshwx.com.cn' +
				':' +
				ws_port +
				'/' +
				'ccws' +
				'?reg=none';

			this.xtalk_configuration = {
				bundlePolicy: 'max-bundle',
				iceServers: [
					{ urls: 'stun:' + ws_host + ':3478' },
					{
						urls: 'turn:' + ws_host + ':3478',
						username: 'hwx',
						credential: 'jshwx123',
					},
				],
			};
		} else {
			ws_host = this.xtalk_device_addr || default_hostname;
			ws_port = this.xtalk_device_port || '8440';

			//var to_peer_str;
			//if (xtalk_xss_to_gb_url != "none")
			//	to_peer_str ='?device=-1?track=-1?gb=' + xtalk_xss_to_gb_url;
			//else
			//	to_peer_str ='?device=' + xtalk_xss_to_device_id +'?track=-1?gb=none';

			ws_url = 'wss://' + ws_host + ':' + ws_port + '/' + 'rtc' + '?media=talk';
		}
		console.info("Connecting",ws_url)
		this.xTalkSetStatus('Connecting to server ' + ws_url);
		this.xTalkWebsocketConnection = new WebSocket(ws_url);

		this.xTalkWebsocketConnection.addEventListener('open', (event: any) => {
			that.xTalkSetStatus('Registering with server');
			that.xTalkSetConnectState('Connected');
		});

		if (this.xtalk_xss_mode) {
			that.xTalkWebsocketConnection.addEventListener(
				'message',
				this.xTalkOnServerMessageXss.bind(this)
			);
		} else {
			this.xTalkWebsocketConnection.addEventListener(
				'message',
				this.xTalkOnServerMessage.bind(this)
			);
		}

		this.xTalkWebsocketConnection.addEventListener(
			'error',
			this.xTalkOnServerError.bind(this)
		);
		this.xTalkWebsocketConnection.addEventListener(
			'close',
			this.xTalkOnServerClose.bind(this)
		);
	}

	xTalkOnRemoteTrack(event: any) {
		const ele = this.xTalkGetAudioElement();
		if (ele) {
			if (ele.srcObject !== event.streams[0]) {
				console.log('xTalkOnRemoteTrack');
				ele.srcObject = event.streams[0];
			}
		}
	}

	xTalkErrorUserMediaHandler() {
		this.xTalkSetError("Browser doesn't support getUserMedia!");
	}

	xTalkCreateCall(msg: any) {
		// Reset connection attempts because we connected successfully
		this.xTalkConnectAttempts = 0;
		let that = this;

		console.log('Creating RTCPeerConnection');

		this.xTalkWebrtcPeerConnection = new RTCPeerConnection(
			this.xtalk_configuration
		);
		this.xTalkWebrtcPeerConnection.ontrack = this.xTalkOnRemoteTrack.bind(this);
		/* Send our video/audio to the other peer */
		this.xTalkLocalStrmPromise = this.xTalkGetLocalStream()
			.then((stream) => {
				console.log('Adding local stream');
				that.xTalkWebrtcPeerConnection.addStream(stream);
				return stream;
			})
			.catch(this.xTalkSetError);

		if (msg != null && !msg.sdp) {
			console.log("WARNING: First message wasn't an SDP message!?");
		}

		this.xTalkWebrtcPeerConnection.onicecandidate = (event: any) => {
			// We have a candidate, send it to the remote party with the
			// same uuid
			if (event.candidate == null) {
				console.log('ICE Candidate was null, done');
				return;
			}

			if (!that.xtalk_xss_mode)
				that.xTalkWebsocketConnection.send(
					JSON.stringify({ ice: event.candidate })
				);
			else
				that.xTalkWebsocketConnection.send(
					'PEER_MSG session_id=' +
						this.xTalkSessionID +
						' ' +
						JSON.stringify({ ice: event.candidate })
				);

			that.xTalkSetStatus('Sending ICE ');
		};

		if (msg != null)
			this.xTalkSetStatus('Created peer connection for call, waiting for SDP');

		return this.xTalkLocalStrmPromise;
	}
}

export { xTalker };
