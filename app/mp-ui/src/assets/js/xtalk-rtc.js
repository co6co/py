var xtalk_xss_mode = false;
var xtalk_constraints = { video: false, audio: true };
var xtalk_configuration;

var xtalk_want_remote_offerer = false;

var xtalk_audio_element_id = 'audiostrm';

var xtalk_device_addr;
var xtalk_device_port;

// Set this to override the automatic detection in xtalk_websocket_server_connect()
var xtalk_xss_to_device_id = -1;
var xtalk_xss_to_gb_url = 'none';
var xtalk_xss_addr;
var xtalk_xss_port;

var xtalk_conn_state = 'Disconnected';

var xTalkWebrtcPeerConnection;
var xTalkWebsocketConnection;
var xTalkConnectAttempts = 0;

var xTalkPeerID;
var xTalkCallID;
var xTalkSessionID;

// Promise for local stream after constraints are approved by the user
var xTalkLocalStrmPromise;

function isNum(txt) {
	if (!isNaN(parseFloat(txt)) && isFinite(txt)) return true;
	else return false;
}

function xTalkSetConnectState(value) {
	xtalk_conn_state = value;
}

function xTalkResetState() {
	// This will call xTalkOnServerClose()
	xtalk_audio_element_id = 'audiostrm';
	xtalk_xss_to_gb_url = 'none';
	xTalkWebsocketConnection.close();
}

function xtalk_websocket_server_disconn() {
	xTalkResetState();
}

function xTalkHandleIncomingError(error) {
	xTalkSetError('ERROR: ' + error);
	xTalkResetState();
}

function xTalkGetAudioElement() {
	return document.getElementById(xtalk_audio_element_id);
}

function xTalkSetStatus(text) {
	console.log(text);
	var span = document.getElementById('status');
	// Don't set the status if it already contains an error
	if (!span) return;
	if (!span.classList.contains('error')) span.textContent = text;
}

function xTalkSetError(text) {
	console.error(text);
	var span = document.getElementById('status');
	if (!span) return;
	span.textContent = text;
	span.classList.add('error');
}

// SDP offer received from peer, set remote description and create an answer
function xTalkoOnIncomingSDP(sdp) {
	xTalkWebrtcPeerConnection
		.setRemoteDescription(sdp)
		.then(() => {
			xTalkSetStatus('Remote SDP set');
			if (sdp.type != 'offer') return;
			xTalkSetStatus('Got SDP offer');
			xTalkLocalStrmPromise
				.then((stream) => {
					xTalkSetStatus('Got local stream, creating answer');
					xTalkWebrtcPeerConnection
						.createAnswer()
						.then(xTalkOnLocalDescription)
						.catch(xTalkSetError);
				})
				.catch(xTalkSetError);
		})
		.catch(xTalkSetError);
}

// Local description was set, send it to peer
function xTalkOnLocalDescription(desc) {
	console.log('Got local description: ' + JSON.stringify(desc));
	xTalkWebrtcPeerConnection.setLocalDescription(desc).then(function () {
		xTalkSetStatus('Sending SDP ' + desc.type);
		sdp = { sdp: xTalkWebrtcPeerConnection.localDescription };

		if (!xtalk_xss_mode) xTalkWebsocketConnection.send(JSON.stringify(sdp));
		else
			xTalkWebsocketConnection.send(
				'PEER_MSG session_id=' + xTalkSessionID + ' ' + JSON.stringify(sdp)
			);
	});
}

function xTalkOnGenerateOffer() {
	xTalkWebrtcPeerConnection
		.createOffer()
		.then(xTalkOnLocalDescription)
		.catch(xTalkSetError);
}

// ICE candidate received from peer, add it to the peer connection
function xTalkOnIncomingICE(ice) {
	var candidate = new RTCIceCandidate(ice);
	xTalkWebrtcPeerConnection.addIceCandidate(candidate).catch(xTalkSetError);
}

function xTalkOnServerMessageXss(event) {
	var msg;
	var msg_splits;
	var sdp_ice_msg;

	console.log('Received ' + event.data);

	msg_splits = event.data.split(' ');
	if (msg_splits.length == 2 && msg_splits[0] == 'REGIST_OK') {
		//REGIST_OK ****
		peer_id = msg_splits[1];

		console.log('Regist to ccws successed: ' + peer_id);

		xTalkCallID = 1;

		var to_peer_str;
		if (xtalk_xss_to_gb_url != 'none')
			to_peer_str = 'device=-1 track=-1 gb=' + xtalk_xss_to_gb_url;
		else to_peer_str = 'device=' + xtalk_xss_to_device_id + ' track=-1 gb=none';

		xTalkWebsocketConnection.send(
			'SESSION call_id=' + xTalkCallID + ' media=talk ' + to_peer_str
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
			xTalkSessionID = ssid_splits[1];

		console.log(
			'session established: ' + xTalkSessionID + ' for call ' + xTalkCallID
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

		if (xTalkSessionID == null || to_ssid != xTalkSessionID) return;

		reason_splits = msg_splits[2].split('=', 2);
		if (reason_splits.length == 2 && reason_splits[0] == 'reason')
			str_reason = reason_splits[1];

		//此处要释放跟session相关的资源
		console.log('session bye: ' + xTalkSessionID + ' reason:' + str_reason);
	} else if (msg_splits.length >= 2 && msg_splits[0] == 'PEER_MSG') {
		//PEER_MSG session_id=* sdp或ice消息串
		var ssid_splits;
		var to_ssid;

		ssid_splits = msg_splits[1].split('=', 2);
		if (ssid_splits.length == 2 && ssid_splits[0] == 'session_id')
			to_ssid = ssid_splits[1];

		if (xTalkSessionID == null || to_ssid != xTalkSessionID) return;

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
		if (!xTalkWebrtcPeerConnection) xTalkCreateCall(msg);

		if (msg.sdp != null) {
			xTalkoOnIncomingSDP(msg.sdp);
		} else if (msg.ice != null) {
			xTalkOnIncomingICE(msg.ice);
		} else {
			xTalkHandleIncomingError('Unknown incoming JSON: ' + msg);
		}
	} else {
		console.log('unexpected message: ' + event.data);
	}
}

function xTalkOnServerMessage(event) {
	console.log('Received ' + event.data);

	if (event.data.startsWith('ERROR')) {
		xTalkHandleIncomingError(event.data);
		return;
	}
	// Handle incoming JSON SDP and ICE messages
	try {
		msg = JSON.parse(event.data);
	} catch (e) {
		if (e instanceof SyntaxError) {
			xTalkHandleIncomingError('Error parsing incoming JSON: ' + event.data);
		} else {
			xTalkHandleIncomingError('Unknown error parsing response: ' + event.data);
		}
		return;
	}

	// Incoming JSON signals the beginning of a call
	if (!xTalkWebrtcPeerConnection) xTalkCreateCall(msg);

	if (msg.sdp != null) {
		xTalkoOnIncomingSDP(msg.sdp);
	} else if (msg.ice != null) {
		xTalkOnIncomingICE(msg.ice);
	} else {
		xTalkHandleIncomingError('Unknown incoming JSON: ' + msg);
	}
}

function xTalkOnServerClose(event) {
	xTalkSetStatus('Disconnected from server');
	//resetVideo();

	if (xTalkWebrtcPeerConnection) {
		xTalkWebrtcPeerConnection.close();
		xTalkWebrtcPeerConnection = null;
	}

	xTalkSetConnectState('Disonnected');

	// Reset after a second
	//window.setTimeout(xtalk_websocket_server_connect, 1000);
}

function xTalkOnServerError(event) {
	xTalkSetError(
		'Unable to connect to server, did you add an exception for the certificate?'
	);
	// Retry after 3 seconds
	//window.setTimeout(xtalk_websocket_server_connect, 3000);
}

function xTalkGetLocalStream() {
	console.log(JSON.stringify(xtalk_constraints));

	// Add local stream
	if (navigator.mediaDevices.getUserMedia) {
		return navigator.mediaDevices.getUserMedia(xtalk_constraints);
	} else {
		xTalkErrorUserMediaHandler();
	}
}

function xtalk_websocket_server_connect() {
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
	if (xtalk_xss_mode) {
		ws_host = xtalk_xss_addr || default_hostname;
		ws_port = xtalk_xss_port || '8440';
		ws_url =
			'wss://' +
			'stream.jshwx.com.cn' +
			':' +
			ws_port +
			'/' +
			'ccws' +
			'?reg=none';

		xtalk_configuration = {
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
		ws_host = xtalk_device_addr || default_hostname;
		ws_port = xtalk_device_port || '8440';

		//var to_peer_str;
		//if (xtalk_xss_to_gb_url != "none")
		//	to_peer_str ='?device=-1?track=-1?gb=' + xtalk_xss_to_gb_url;
		//else
		//	to_peer_str ='?device=' + xtalk_xss_to_device_id +'?track=-1?gb=none';

		ws_url = 'wss://' + ws_host + ':' + ws_port + '/' + 'rtc' + '?media=talk';
	}

	xTalkSetStatus('Connecting to server ' + ws_url);
	xTalkWebsocketConnection = new WebSocket(ws_url);

	xTalkWebsocketConnection.addEventListener('open', (event) => {
		xTalkSetStatus('Registering with server');
		xTalkSetConnectState('Connected');
	});

	if (xtalk_xss_mode) {
		xTalkWebsocketConnection.addEventListener(
			'message',
			xTalkOnServerMessageXss
		);
	} else {
		xTalkWebsocketConnection.addEventListener('message', xTalkOnServerMessage);
	}

	xTalkWebsocketConnection.addEventListener('error', xTalkOnServerError);
	xTalkWebsocketConnection.addEventListener('close', xTalkOnServerClose);
}

function xTalkOnRemoteTrack(event) {
	if (xTalkGetAudioElement().srcObject !== event.streams[0]) {
		console.log('xTalkOnRemoteTrack');
		xTalkGetAudioElement().srcObject = event.streams[0];
	}
}

function xTalkErrorUserMediaHandler() {
	xTalkSetError("Browser doesn't support getUserMedia!");
}

function xTalkCreateCall(msg) {
	// Reset connection attempts because we connected successfully
	xTalkConnectAttempts = 0;

	console.log('Creating RTCPeerConnection');

	xTalkWebrtcPeerConnection = new RTCPeerConnection(xtalk_configuration);
	xTalkWebrtcPeerConnection.ontrack = xTalkOnRemoteTrack;

	/* Send our video/audio to the other peer */
	xTalkLocalStrmPromise = xTalkGetLocalStream()
		.then((stream) => {
			console.log('Adding local stream');
			xTalkWebrtcPeerConnection.addStream(stream);
			return stream;
		})
		.catch(xTalkSetError);

	if (msg != null && !msg.sdp) {
		console.log("WARNING: First message wasn't an SDP message!?");
	}

	xTalkWebrtcPeerConnection.onicecandidate = (event) => {
		// We have a candidate, send it to the remote party with the
		// same uuid
		if (event.candidate == null) {
			console.log('ICE Candidate was null, done');
			return;
		}

		if (!xtalk_xss_mode)
			xTalkWebsocketConnection.send(JSON.stringify({ ice: event.candidate }));
		else
			xTalkWebsocketConnection.send(
				'PEER_MSG session_id=' +
					xTalkSessionID +
					' ' +
					JSON.stringify({ ice: event.candidate })
			);

		xTalkSetStatus('Sending ICE ');
	};

	if (msg != null)
		xTalkSetStatus('Created peer connection for call, waiting for SDP');

	return xTalkLocalStrmPromise;
}

export {
	xtalk_xss_mode,
	xtalk_xss_addr,
	xtalk_xss_port,
	xtalk_audio_element_id,
	xtalk_conn_state,
	xtalk_xss_to_device_id,
	xtalk_xss_to_gb_url,
	xtalk_websocket_server_disconn,
	xtalk_websocket_server_connect,
	isNum,
}
