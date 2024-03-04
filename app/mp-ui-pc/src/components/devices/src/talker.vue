<template>
	<audio id="audiostrm" autoplay></audio>
</template>
<script setup lang="ts">
	import { default as mqtt } from 'mqtt';
	import {
		ref,
		watch,
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
	import { xTalker } from '../xtalk';
	import { useAppDataStore } from '../../../store/appStore';
	import { type talkState ,type talkerMessageData} from './types';
 

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
	const emit = defineEmits<{ 
		(event: 'stateChange', data: talkState): void ,
		(event: 'log', data: string): void
		(event: 'onMessage', data: any):void }>();
	talker.xTalkSetStatus=(value: string)=>{ 
		emit("log",value)
	}
	talker.xTalkSetError=(value: string)=>{
		console.warn("error",value)
		emit("log",value)
	}
	talker.xTalkSetConnectState = (value: string) => {
		talker.xtalk_conn_state = value;
		let connect = 0;
		console.info('talkerStateChange：', value);
		if (value == 'Connected') connect = 1;
		else connect = 0;
		emit('stateChange', {
			state: connect,
			stateDesc: value,
			talkNo: talker.xtalk_xss_to_device_id 
		});
	};
	talker.onMessage = (value: talkerMessageData) => {  
		emit('onMessage', value);
	};
	watch(
		() => props.talkNo,
		(n, o) => {
			if (n) {
				let peer = n; 
				if (typeof peer == 'number') talker.xtalk_xss_to_device_id = peer;
				else if (typeof peer == 'string') talker.xtalk_xss_to_gb_url = peer;
				if(connected.value)disconnect()
				else talker.xTalkSetConnectState("Disonnected")
			}
		}
	);
	const connected = ref(false);
	const connect = () => {
		talker.xtalk_xss_mode = true;
		talker.xtalk_xss_addr = store.data.xssConfig.ip;
		talker.xtalk_xss_port = store.data.xssConfig.port;
		talker.xtalk_audio_element_id = 'audiostrm';

		//取对端标识，设备ID或者GB URL形式

		talker.xtalk_websocket_server_connect();
		connected.value = true;
	};
	const disconnect = () => {
		talker.xtalk_websocket_server_disconn();
		//console.info('call disconnect,state:', talker.xtalk_conn_state);
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
