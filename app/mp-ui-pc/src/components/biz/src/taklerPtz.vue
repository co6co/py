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
		<el-text class="mx-1">
			状态:
			<el-tag
				class="ml-2"
				:type="talkerState.data.state == 1 ? 'success' : 'info'"
				>{{ talkerState.data.stateDesc }}</el-tag
			>
		</el-text>
		<el-text class="mx-1">
			对讲号:
			<el-tag
				class="ml-2"
				:type="talkerState.data.state == 1 ? 'success' : 'info'"
				>{{ talkerState.data.talkNo }}</el-tag
			>
		</el-text>
	</div>
	<div>
		<el-text class="mx-1"> 信息: </el-text>
		<el-row>
			<el-tooltip   v-for="(item,index) in takerLogInfo">
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
	let mqttInit = false;
	try {
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
		mqttInit = true;
	} catch (e) {
		ElMessage.error(`连接到MQTT服务器失败:${e}`);
	}

	function unique(arr: Array<mqttMessage>) {
		const res = new Map();
		return arr.filter((a) => !res.has(a.UUID) && res.set(a.UUID, 1));
	}
	const OnPtz = (name: p.ptz_name, type: p.ptz_type) => {
		let param = {
			payload: {
				BoardId: 'RJ-BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60',
				Event: '/app_network_query_v2',
			},
			qos: 0,
			retain: false,
		};
		let ptzcmd = 'A50F010800FA00B7';
		if (!mqttInit) {
			ElMessage.warning('MQtt 服务 未连接！');
			return;
		}
		if (type == 'starting') {
			switch (name) {
				case 'up':
					ptzcmd = 'A50F010800FA00B7';
					break;
				case 'down':
					ptzcmd = 'A50F010400FA00B3';
					break;
				case 'right':
					ptzcmd = 'A50F0101FA0000B0';
					break;
				case 'left':
					ptzcmd = 'A50F0102FA0000B1';
					break;
				case 'zoomin':
					ptzcmd = 'A50F01200000A075';
					break;
				case 'zoomout':
					ptzcmd = 'A50F01100000A065';
					break;
			}
		} else {
			ptzcmd = 'A50F0100000000B5';
		}
		let sip = props.currentDeviceData?.sip;
		let sn = new Date().getMilliseconds();
		let xml = `
		<?xml version="1.0" encoding="UTF-8"?>
		<Control>
			<CmdType>DeviceControl</CmdType>
			<SN>${sn}</SN>
			<DeviceID>${sip}</DeviceID>
			<PTZCmd>${ptzcmd}</PTZCmd>
		</Control> 
		`;
		console.info('发送', xml);
		Ref_Mqtt.value?.publish('/MANSCDP_cmd', xml);
	};
</script>
