import { default as mqtt } from 'mqtt';
import {
	ref,
	reactive,
	onUnmounted,
	defineComponent,
	watchEffect,
	defineExpose,
} from 'vue';

import { get_xss_config_svc } from '../../api/server/index.js';
//对讲模块
import '../../assets/js/adapter-latest.js';
//import * as r from  '../assets/js/xtalk-rtc.js';
import { xTalker } from '../../components/devices/xtalk.js';

const xss_config = await get_xss_config_svc();
if (xss_config.code != 0)
	console.warn('获取xss服务配置失败：', xss_config.message);

export const talker = defineComponent({
	name: 'talker',
	props: {
		talkNo: {
			type: [Number, String],
		},
	},
	setup(props) {
		let talker = new xTalker();
		talker.xtalk_xss_mode = true;
		talker.xtalk_xss_addr = xss_config.data.ip;
		talker.xtalk_xss_addr = xss_config.data.ip;
		talker.xtalk_xss_port = xss_config.data.port;
		talker.xtalk_audio_element_id = 'audiostrm';
		const connected = ref(false);
		const connect = () => {
			//取对端标识，设备ID或者GB URL形式
			let peer = props.talkNo;
			if (typeof peer == 'number') talker.xtalk_xss_to_device_id = peer;
			else if (typeof peer == 'string') talker.xtalk_xss_to_gb_url = peer;
			talker.xtalk_websocket_server_connect();
			console.info(talker.xtalk_conn_state);
			connected.value = true;
		};
		const disconnect = () => {
			talker.xtalk_websocket_server_disconn();
			console.info(talker.xtalk_conn_state);
			connected.value = false;
		};
		//示例方法 不在使用
		function onConnectClicked() {
			if (talker.xtalk_conn_state == 'Connected') {
				talker.xtalk_websocket_server_disconn();
				connected.value = true;
				console.info(talker.xtalk_conn_state);
			} else {
				//取对端标识，设备ID或者GB URL形式
				let peer = props.talkNo;
				if (typeof peer == 'number') talker.xtalk_xss_to_device_id = peer;
				else if (typeof peer == 'string') talker.xtalk_xss_to_gb_url = peer;
				talker.xtalk_websocket_server_connect();
				console.info(talker.xtalk_conn_state);
			}
		}

		defineExpose({
			talker,
			connect,
			disconnect,
		}); 
		return () => { 
			return (<audio id="audiostrm" autoplay></audio>)
		};
	},
	methods: {},
});
