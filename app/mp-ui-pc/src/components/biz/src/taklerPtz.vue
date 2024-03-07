<template>
	<talker
		ref="talkerRef"
		@state-change="onTalkerChange"
		@log="onTalkerLog"
		@on-message="onMessage"
		:talk-no="talkbackNo"></talker>
	<ptz
		@ptz="OnPtz"
		@center-click="onTalker"
		:taker-state="allowTakler"
		:ptz-enable="ptzEnable"
		:taker-enable="talkerEnable">
	</ptz>
	<div id="stateInfo">
		<el-container>
			<div style="height: 100%;">
				<div style="margin: 36px 0 5px 0">
					<el-text class="mx-1" tag="p">
						对讲用户:{{talkerState.userName}}
					</el-text>
					<el-text class="mx-1" tag="p">
						xss状态:
						<el-tag
							class="ml-2"
							:type="talkerState.data.state == 1 ? 'success' : 'info'"
							>{{ talkerState.data.stateDesc }}</el-tag
						> 
						
					</el-text>
					<el-text class="mx-1" tag="p">
						对讲设备状态:
						<el-tag
							class="ml-2"
							:type="(!talkerState.busy&&talkerState.online||talkerState.busy&&talkerState.online&&talkerState.isCurrentUser)  ? 'success' : 'info'"
							>{{ taklerDesc}} </el-tag
						> 
					</el-text>

					<el-text class="mx-1" tag="p">
						对讲号:
						<el-tag
							class="ml-2"
							:type="talkerState.data.state == 1 ? 'success' : 'info'"
							>{{ talkerState.data.talkNo }}</el-tag> 
					</el-text>
					<el-text class="mx-1" tag="p">
						Mqtt状态:
						<el-tag class="ml-2" :type="mqttConneted ? 'success' : 'info'">{{
							mqttConneted ? '已连接' : '未连接'
						}}</el-tag>
					</el-text>
					<el-text class="mx-1" tag="p">
						mqtt消息:
						<el-tag
							class="ml-2"
							:type="messangeSendRef.send ? 'success' : 'info'"
							>{{
								messangeSendRef.send
									? '已发送'
									: `未发送:${messangeSendRef.length}`
							}}</el-tag
						>
					</el-text>
				</div>
				<div>
					<el-text class="mx-1"> 信息: </el-text>
					<el-row>
						<el-tooltip :key="index" v-for="(item, index) in takerLogInfo">
							<template #content>
								{{ item }}
							</template>
							<el-text truncated>
								<el-tag>{{ index + 1 }}</el-tag> {{ item }}
							</el-text>
						</el-tooltip>
					</el-row>
				</div>
			</div>
		</el-container>
	</div>
</template>
<script setup lang="ts">
	import { PropType, ref, watch, reactive, computed,nextTick } from 'vue';
	import { useMqtt, mqqt_server } from '../../../utils/mqtting';
	import { talker, types as dType } from '../../../components/devices';
	import { ptz as cmd  } from '../../../components/biz';
	import { ptz } from '../../../components/stream';
	import * as p from '../../../components/stream/src/types/ptz';

	import { ElMessage } from 'element-plus';
	import {RtcOnlineState,RtcSessionState} from '../../../components/devices/gb28181';
	import {type gbTaklerOnlineList,type gbTaklerOnlineSessionList} from '../../../api/deviceState';
 

	 
	const props = defineProps({
		currentDeviceData: {
			type: Object as PropType<dType.DeviceData>,
		},
	});
	const talkbackNo = computed(() => {
		if (props.currentDeviceData) {
			return props.currentDeviceData.talkbackNo;
		}
		return -1;
	});
	//正在对讲状态
	const allowTakler=computed(()=>{
		//对讲服务器在线 
		//对讲设备在线
		if(talkerState.data.state == 1&&talkerState.online&&talkerState.busy&&talkerState.isCurrentUser)return true
		return false;
		
	})
	const taklerDesc=computed(()=>{
		if(!talkerState.online){return "不在线"}
		if(talkerState.isCurrentUser&&talkerState.busy){return "对讲中..."}
		if(talkerState.busy){return "设备忙"} 
		return "空闲"
	})
	watch(
		() => props.currentDeviceData,
		(n, o) => {
			takerLogInfo.value = [];
			if (props.currentDeviceData){
				console.info("changeed",props.currentDeviceData.talkbackNo)
				RtcOnlineState(onDeviceOnline)
				RtcSessionState(onDeviceSession)
			}
			
		},{deep:true}
	);
 
	const talkerState = reactive<{ data: dType.talkState }&{online:boolean;busy:boolean,userName:string}&{sessionId:string,isCurrentUser:boolean}>({
		data: { state: -1, stateDesc: '', talkNo: -1 },
		online:false,busy:true,userName:"",
		sessionId:"",isCurrentUser:false,
	});
	//gb 设备在线/空闲 
	const onDeviceOnline=(data:gbTaklerOnlineList)=>{
		let f=false
		if(props.currentDeviceData&&props.currentDeviceData.talkbackNo>-1&&data&&data.count>0){ 
			for(let i=0;i<data.data.length;i++){
				let item=data.data[i];
				if(item['device-id']==props.currentDeviceData.talkbackNo){
					talkerState.online=true;
					f=true;
					break
				}
			}
		}
		if(!f)talkerState.online=false;
	}
	//设备忙或者空闲
	const onDeviceSession=(data:gbTaklerOnlineSessionList)=>{  
		if(props.currentDeviceData&&props.currentDeviceData.talkbackNo>-1&&data){
			let f=false
			for(let i=0;i<data.data.length;i++){
				let item=data.data[i];
				if(item['call-info'].device==props.currentDeviceData.talkbackNo){
					talkerState.busy=true;
					talkerState.userName=item['call-info'].from_userid
					//当前会话
					if(talkerState.sessionId==item['session-id']){  
						talkerState.isCurrentUser=true
					}else{
						talkerState.isCurrentUser=false
					}
					f=true
					break
				}
			}
			//不在列表中
			if (!f){
				talkerState.userName=""
				talkerState.busy=false;
				talkerState.isCurrentUser=false
			}
		}
	} 
	const talkerRef = ref();
	
	const takerLogInfo = ref<string[]>([]);
	const onTalkerChange = (state: dType.talkState) => {
		talkerState.data = state;
	};
	const onTalkerLog = (log: string) => {
		if (takerLogInfo.value.length > 9) takerLogInfo.value = [];
		takerLogInfo.value.push(log);
	};
	const onTalker = (active: boolean) => {
		let f=false
		if (active&&!talkerState.busy&&talkerState.online)f=true, talkerRef.value.connect();
		else if(!active&&talkerState.isCurrentUser)   f=true,talkerRef.value.disconnect();
		if(f){
			nextTick(()=>{
				//更新状态
				RtcSessionState(onDeviceSession)
			}) 
		} 
	};
	const onMessage=(data:dType.talkerMessageData)=>{ 
		talkerState.sessionId=data.SessionId 
	} 
	const ptzEnable = computed(() => {
		if (props.currentDeviceData && props.currentDeviceData.sip) {
			return true;
		}
		return false;
	});
	const talkerEnable = computed(() => {
		if (props.currentDeviceData && props.currentDeviceData.talkbackNo) {
			return true;
		}
		return false;
	});
	//piz
	const { startMqtt, Ref_Mqtt } = useMqtt();
	interface mqttMessage {
		UUID?: string;
	}
	let arr: Array<mqttMessage> = [];
	const mqttConneted = ref(false);
	const messangeSendRef = ref<{ send: boolean; length: number }>({
		send: true,
		length: 0,
	});
	const setMessageState = () => {
		messangeSendRef.value.send = Ref_Mqtt.value?.client.queue.length == 0;
		if (Ref_Mqtt.value)
			messangeSendRef.value.length = Ref_Mqtt.value?.client.queue.length;
	};
	const mqttConnectBck = (connected: boolean, error: any) => {
		mqttConneted.value = connected;
	};
	try {
		startMqtt(
			mqqt_server,
			'/edge_app_controller_reply',
			(topic: any, message: any) => {
				const msg: mqttMessage = JSON.parse(message.toString());
				arr.unshift(msg); //新增到数组起始位置
				let data = unique(arr);
				//alert(JSON.stringify(data));
				console.warn(unique(arr));
			},
			mqttConnectBck
		);
	} catch (e) {
		ElMessage.error(`连接到MQTT服务器失败:${e}`);
	}
	function unique(arr: Array<mqttMessage>) {
		const res = new Map();
		return arr.filter((a) => !res.has(a.UUID) && res.set(a.UUID, 1));
	} 
	const OnPtz = (name: p.ptz_name, type: p.ptz_type, speed: number) => {
		if (!mqttConneted.value) {
			ElMessage.warning('MQtt 服务 未连接！');
			messangeSendRef.value.send = false;
			return;
		}
		if (props.currentDeviceData && props.currentDeviceData.sip) {
			//let strCmd=cmd.createPtzCmd(speed,type,name)
			//cmd.testPtzCmdStr("A50F0100000000B5") 
			let obj = cmd.generatePtzXml(
				props.currentDeviceData?.sip,
				speed,
				type,
				name
			);
			onTalkerLog(`ptz:${name}->${type},speed:${speed},sn:${obj.sn},`);
			console.info('发送PTZ命令：', obj.xml); 
			const client = Ref_Mqtt.value?.publish(
				'/MANSCDP_cmd',
				obj.xml,
				(err, pkg) => {
					setMessageState();
				}
			);
			setMessageState();
		}
	};
</script>

<style lang="less" scoped>
	#stateInfo { 
		margin-top: 40px; 
	}
</style>
