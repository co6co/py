<template>
	<audio id="audiostrm" autoplay></audio>
</template>
<script setup lang="ts">
	import { default as mqtt } from 'mqtt';
	import {
		ref,
		reactive,
		onBeforeMount,
		onMounted,
		onUnmounted,
		defineComponent,
		watchEffect,
		defineExpose,
	} from 'vue';

	import { get_xss_config_svc } from '../../../api/server/index.js';
	//对讲模块
	import '../../../assets/js/adapter-latest.js';
	//import * as r from  '../assets/js/xtalk-rtc.js';
	import { xTalker } from '../../../utils/xtalk';
	import { useAppDataStore } from '../../../store/appStore';

	const props = defineProps({
		ip: {
			type: String,
		},
		port: {
			type: Number,
		},
		talkNo: {
			type: [Number, String],
		},
	});
	const store = useAppDataStore();

	onBeforeMount(async () => {
		try { 
			await store.setXssConfig();
		} catch (e) {
			console.error(e);
		}
	});
	onMounted(() => {}); 
	let talker = new xTalker(); 
	const connected = ref(false);
	const connect = () => {
		talker.xtalk_xss_mode = true;
		talker.xtalk_xss_addr = store.data.xssConfig.ip;
		talker.xtalk_xss_port = store.data.xssConfig.port;
		talker.xtalk_audio_element_id = 'audiostrm';

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
</script>
