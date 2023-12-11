 import  {  default as mqtt} from "mqtt"
//const mqtt = require('mqtt/dist/mqtt'); //#CommonJS 模块中  tsconfig.json 文件中可以设置 module 选项来
import type { MqttClient, OnMessageCallback } from 'mqtt'; 
import { ref, onUnmounted } from 'vue';
 
class MQTTing {
    url: string; // mqtt地址
    topics: string[]
    client!: MqttClient;
    constructor( url: string,topics:string[]) { 
        // 虽然是mqtt但是在客户端这里必须采用websock的链接方式
        //this.url = 'ws://xxx.xxx.xxx.x:xxxx/mqtt';
        this.url = url;
        this.topics=topics
    }
    //初始化mqtt
    init() {
        const options = {
            clean: true,// 保留会话
            // 认证信息
            clientId: "clientId",
            username: "admin", //用户名 不用的话什么也不用填
            password: "admin", //密码 不用的话什么也不用填
            connectTimeout: 4000, // 超时时间
            reconnectPeriod: 4000, // 重连时间间隔
        };
        
        this.client = mqtt.connect(this.url, options);
        this.client.on('error', (error: any) => {
            console.log(error);
        });
        this.client.on('reconnect', ( ) => {
            
        });
    } 

    subscribe(topic?:string) {
        if (topic!=null) this.topics.push(topic)
        this.topics.forEach((item) => {
            this.client.on('connect', () => {
                this.client.subscribe(item, (error: any) => {
                    if (!error) {
                        console.log('订阅成功');
                    } else {
                        console.log('订阅失败');
                    }
                });
            });
        })
    } 
    getMessage  (callback: OnMessageCallback)   {  
        this.client.on('message', callback);
    }
    //取消订阅
    unsubscribes( ) {
       this.topics.forEach((topic) => {
            this.client.unsubscribe( topic, (error?: Error) => {
                if (!error) {
                    console.log(topic , '取消订阅成功');
                } else {
                    console.log(topic, '取消订阅失败');
                }
            });
        })
    } 
    publish (topic:string,message:string|Buffer){ 
       this.client.publish(topic,message );
    } 
    //结束链接
    close() {
        this.client.end();
    }
}
 
 /**
 * mqtt 使用
 */
  function useMqtt() {
    const Ref_Mqtt = ref<MQTTing | null>(null); 
    const startMqtt = (url:string, topic: string, callback: OnMessageCallback) => {
        //设置订阅地址
        Ref_Mqtt.value = new MQTTing(url ,[topic]);
        console.log(Ref_Mqtt.value)
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
        startMqtt,Ref_Mqtt
    };
}

 
/**
 * 长连接使用
 */
function useMqtting() {
    let topicData: string[];
    const Ref_Mqtt = ref<MQTTing | null>(null);
    const startMqtt = (url:string, topics: string[], callback: OnMessageCallback) => {
        topicData = topics;
        //设置订阅地址
        Ref_Mqtt.value = new MQTTing(url,topics);
        //初始化mqtt
        Ref_Mqtt.value.init();
        Ref_Mqtt.value?.subscribe( );
        getMessage(callback);
        
    };

    const getMessage = (callback: OnMessageCallback) => {
        // PublicMqtt.value?.client.on('message', callback);
        Ref_Mqtt.value?.getMessage(callback)
    };
    onUnmounted(() => {
        //页面销毁结束订阅
        if (Ref_Mqtt.value) {
            Ref_Mqtt.value.unsubscribes();
            Ref_Mqtt.value.close();
        }
    });
 
    return {
        startMqtt,PublicMqtt: Ref_Mqtt
    };
}


export {useMqtt,useMqtting,MQTTing}
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