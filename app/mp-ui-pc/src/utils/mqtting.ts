import { NONAME } from 'dns';
import { connect } from 'http2';
import { default as mqtt } from 'mqtt';
//const mqtt = require('mqtt/dist/mqtt'); //#CommonJS 模块中  tsconfig.json 文件中可以设置 module 选项来
import type { MqttClient, OnMessageCallback,PacketCallback } from 'mqtt';
import { ref, onUnmounted } from 'vue';

const mqqt_server = import.meta.env.VITE_mqtt_server;
const debug = Boolean(Number(import.meta.env.VITE_IS_DEBUG));

type ConnectBck=(connected:boolean, error?:any)=>void
class MQTTing {
	url: string; // mqtt地址
	topics: string[];
	client!: MqttClient;
	connectBck?:ConnectBck
	constructor(url: string, topics: string[],onConnectBck?:ConnectBck) {
		// 虽然是mqtt但是在客户端这里必须采用websock的链接方式
		//this.url = 'ws://xxx.xxx.xxx.x:xxxx/mqtt';
		this.url = url;
		this.topics = topics;
		this.connectBck=onConnectBck
	}
	//初始化mqtt
	init() {
		const options = {
			clean: true, // 保留会话
			// 认证信息
			clientId: 'mqttjs_' + Math.random().toString(16),
			username: 'mqttroot', //用户名 不用的话什么也不用填
			password: 'hQEMA4fLSGZcsDhlAQf', //密码 不用的话什么也不用填
			connectTimeout: 50*1000, // 超时时间
			reconnectPeriod:1000, // 两次重新连接之间的间隔，客户端 ID 重复、认证失败等客户端会重新连接；
			keepalive:60 , //心跳 
		};
		let that=this;
		this.client = mqtt.connect(this.url, options); 
		this.client.on('error', (error: any) => {
			console.log(error);
			if(that.connectBck)that.connectBck(false,error )
		});
		this.client.on('reconnect', () => {
			if(that.connectBck)that.connectBck(false,"重新连接..." )
			console.warn(new Date(), "重试重新连接....")
		});
	}
	subscribe(topic?: string) {
		let that=this;
		if (topic != null) this.topics.push(topic);
		this.topics.forEach((item) => {
			this.client.on('connect', () => {
				this.client.subscribe(item, (error: any) => {
					if (!error) {
						if(that.connectBck)that.connectBck(true )
						console.log('订阅成功');
					} else {
						if(that.connectBck)that.connectBck(false )
						console.log('订阅失败');
					}
				});
			});
		});
	}
	getMessage(callback: OnMessageCallback) {
		this.client.on('message', callback);
	}
	//取消订阅
	unsubscribes() {
		this.topics.forEach((topic) => {
			this.client.unsubscribe(topic, (error?: Error) => {
				if (!error) {
					console.log(topic, '取消订阅成功');
				} else {
					console.log(topic, '取消订阅失败');
				}
			});
		});
	}
	publish(topic: string, message: string | Buffer,bck?:PacketCallback) {
		return this.client.publish(topic, message,bck);
	}
	//结束链接
	close() {
		this.client.end();
	}
}
const getUrl = (url: string) => {
	let urlFull = url;
	if (!debug) {
		let protocol = 'wss:';
		if (window.location.protocol == 'http:') protocol = 'wss:';
		urlFull = `${protocol}//${window.location.host}${url}`;
	}
    return urlFull;
};
/**
 * mqtt 使用
 */
function useMqtt() {
	const Ref_Mqtt = ref<MQTTing | null>(null);
	const startMqtt = (
		url: string,
		topic: string,
		callback: OnMessageCallback,
		connectbck?:ConnectBck
	) => { 
		//设置订阅地址
		//"tcp://36.137.74.204:9101/mqtt"
		Ref_Mqtt.value = new MQTTing(getUrl(url), [topic],connectbck); 
		//初始化mqtt
		Ref_Mqtt.value.init();
		//链接mqtt
		Ref_Mqtt.value.subscribe();
		getMessage(callback);
	};
	const getMessage = (callback: OnMessageCallback) => {
		Ref_Mqtt.value?.getMessage(callback);
	};
	onUnmounted(() => {
		//页面销毁结束订阅
		if (Ref_Mqtt.value) {
			Ref_Mqtt.value.unsubscribes();
			Ref_Mqtt.value.close();
		}
	});

	return {
		startMqtt,
		Ref_Mqtt,
	};
}

/**
 * 长连接使用
 */
function useMqtting() {
	let topicData: string[];
	const Ref_Mqtt = ref<MQTTing | null>(null);
	const startMqtt = (
		url: string,
		topics: string[],
		callback: OnMessageCallback
	) => {
		topicData = topics;
		//设置订阅地址
		Ref_Mqtt.value = new MQTTing(getUrl(url), topics);
		//初始化mqtt
		Ref_Mqtt.value.init();
		Ref_Mqtt.value?.subscribe();
		getMessage(callback);
	};

	const getMessage = (callback: OnMessageCallback) => {
		// PublicMqtt.value?.client.on('message', callback);
		Ref_Mqtt.value?.getMessage(callback);
	};
	onUnmounted(() => {
		//页面销毁结束订阅
		if (Ref_Mqtt.value) {
			Ref_Mqtt.value.unsubscribes();
			Ref_Mqtt.value.close();
		}
	});

	return {
		startMqtt,
		PublicMqtt: Ref_Mqtt,
	};
}

export { useMqtt, useMqtting, MQTTing, mqqt_server };
/**
 * 
 * 
 * // 使用
// import {useMqtt} from '@/utils/mqtt';
// const { startMqtt } = useMqtt();
// startMqtt('主题topic', (topic, message) => {
//    const msg = JSON.parse(message.toString());
//    console.log(msg);
// });
 

 * 
 * useMqtting
 * 
 * <script lang="ts" setup>
// 自我封装
import {userMqtting} from "@/utils/mqtt";
const { startMqtt } = useMqtt();
 
let arr = [];
startMqtt(
  ["地址1", "地址2", "地址3"],
  (topic: any, message: any) => {
    const msg = JSON.parse(message.toString());
    arr.unshift(msg);
    console.log(unique(arr))
  }
);
//这里我每秒都会接收到数据 进行数据去重
function unique(arr) {
  const res = new Map();
  return arr.filter((a) => !res.has(a.UUID) && res.set(a.UUID, 1));
}
</script> 
 * 
 */
