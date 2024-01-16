<template>
	<talker
		ref="talkerRef"
		@state-change="onTalkerChange"
		@log="onTalkerLog"
		:talk-no="talkbackNo"></talker>
	<ptz
		@ptz="OnPtz"
		@center-click="onTalker"
		:taker-state="talkerState.data.state == 1"
		:ptz-enable="ptzEnable"
		:taker-enable="talkerEnable">
	</ptz>
	<div style="margin: 36px 0 5px 0">

		<el-text class="mx-1" tag="p" >
			xss状态:
			<el-tag
				class="ml-2"
				:type="talkerState.data.state == 1 ? 'success' : 'info'"
				>{{ talkerState.data.stateDesc }}</el-tag
			> 
		</el-text>
		<el-text class="mx-1" tag="p" >
			对讲号:
			<el-tag
				class="ml-2"
				:type="talkerState.data.state == 1 ? 'success' : 'info'"
				>{{ talkerState.data.talkNo }}</el-tag
			>
		</el-text>
		<el-text class="mx-1" tag="p" >
			Mqtt状态:
			<el-tag  
				class="ml-2"
				:type="mqttConneted? 'success' : 'info'"
				>{{ mqttConneted?"连接":"未连接" }}</el-tag
			>
		</el-text>
		
	</div>
	<div>
		<el-text class="mx-1"> 信息: </el-text>
		<el-row>
			<el-tooltip :key="index"   v-for="(item,index) in takerLogInfo">
                <template #content>
                    {{item}}
                </template>
				<el-text truncated>
					<el-tag>{{index+1}}</el-tag> {{ item }}
				</el-text>
			</el-tooltip> 
		</el-row>
	</div>
</template>
<script setup lang="ts">
	import { PropType, ref,watch, reactive, computed } from 'vue';
	import { useMqtt, mqqt_server } from '../../../utils/mqtting';
	import { talker, types as dType } from '../../../components/devices';
	import { ptz as cmd} from '../../../components/biz';
	import { ptz } from '../../../components/stream';
	import * as p from '../../../components/stream/src/types/ptz';

	import { ElMessage } from 'element-plus'; 


	const props = defineProps({
		currentDeviceData: {
			type: Object as PropType<dType.deviceItem>,
		},
	});
	const talkbackNo=computed(()=>{
		if(props.currentDeviceData){
			return props.currentDeviceData.talkbackNo
		}
		return -1
	})
    watch(()=>props.currentDeviceData,(n,o)=>{
        takerLogInfo.value=[]
    })
	const talkerRef = ref();
	const talkerState = reactive<{ data: dType.talkState }>({
		data: { state: -1, stateDesc: '', talkNo: -1 },
	});
	const takerLogInfo = ref<string[]>([]);
	const onTalkerChange = (state: dType.talkState) => {
		talkerState.data = state;
	};
	const onTalkerLog = (log: string) => {
		if (takerLogInfo.value.length > 9) takerLogInfo.value = [];
		takerLogInfo.value.push(log);
	};
	const onTalker = (active: boolean) => {
		if (active) talkerRef.value.connect();
		else {
			talkerRef.value.disconnect();
		}
	};
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
	let mqttConneted = ref(false);
	try {
		console.info(mqqt_server)
		startMqtt(
			mqqt_server,
			'/edge_app_controller_reply',
			(topic: any, message: any) => {
				const msg: mqttMessage = JSON.parse(message.toString());
				arr.unshift(msg); //新增到数组起始位置
				let data = unique(arr);
				alert(JSON.stringify(data));
				console.warn(unique(arr));
			}
		);
		mqttConneted .value= true;
	} catch (e) {
		ElMessage.error(`连接到MQTT服务器失败:${e}`);
	}

	function unique(arr: Array<mqttMessage>) {
		const res = new Map();
		return arr.filter((a) => !res.has(a.UUID) && res.set(a.UUID, 1));
	}
	const OnPtz = (name: p.ptz_name,type: p.ptz_type,speed:number) => { 
		if (!mqttConneted .value) {
			ElMessage.warning('MQtt 服务 未连接！');
			return;
		}
		if (props.currentDeviceData&&props.currentDeviceData.sip){ 
			//let strCmd=cmd.createPtzCmd(speed,type,name)
			//cmd.testPtzCmdStr("A50F0100000000B5")
			onTalkerLog(`ptz:${name}->${type},speed:${speed}`)
			let str= cmd.generatePtzXml(props.currentDeviceData?.sip,speed,type,name)
			console.info("发送PTZ命令：", str)
			let client= Ref_Mqtt.value?.publish('/MANSCDP_cmd', str); 
			console.info(client)
		}
	};
</script>
