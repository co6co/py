<template>
	<el-container>
		<el-header>
			<nav-bar left-text="返回" left-arrow :title="title" @click="onToList()"
		/></el-header>

		<el-main class="ui">
			<stream :sources="player.sources"></stream>
		</el-main>
		<el-footer>
			<notice-bar left-icon="volume-o" :text="noticeMessage" />
			<div class="ptzContent">
				<ptz @ptz="OnPtz"></ptz>
			</div>
		</el-footer>
	</el-container>
	<div></div>
</template>
<script setup lang="ts" name="basetable">
	import {
		ref,
		watch,
		reactive,
		watchEffect,
		nextTick,
		PropType,
		onMounted,
		onBeforeUnmount,
		computed,
	} from 'vue';

	import {
		Image as VanImage,
		Divider,
		List,
		Search,
		Icon,
		Grid,
		GridItem,
		Card,
		NoticeBar,
		Skeleton,
		NavBar,
	} from 'vant';
	import * as api from '../api/device';
	import { stream, ptz } from '../components/stream';
	import { useMqtt, mqqt_server } from '../utils/mqtting';

	import * as d from '../store/types/devices';

	import { useRouter } from 'vue-router';
	import { useAppDataStore } from '../store/appStore';
	const loading = ref(true);
	const dataStore = useAppDataStore();
	const router = useRouter();
	const onToList = () => {
		router.back();
	};
	const title = ref();
	onMounted(() => {
		const rowData = <d.dataItem>dataStore.getState();
		console.info('123', rowData);
		if (!rowData.id) console.warn('数据为加载'), router.back();
		loadData(rowData);
		loading.value = false;
	});
	interface preview_module {
		currentItem?: d.dataItem;
	}
	const preview_module = reactive<preview_module>({});
	/** 播放器 */
	interface player_sources {
		sources: Array<stream_source>;
	}
	const player = reactive<player_sources>({ sources: [] });
	const loadData = (item: d.dataItem) => {
		title.value = item.name + '预览';
		preview_module.currentItem = item;
		if (item.streams) player.sources = item.streams;
	};
	/** ptz */
	const { startMqtt, Ref_Mqtt } = useMqtt();
	interface mqttMessage {
		UUID?: string;
	}
	const noticeMessage = ref('');
	let arr: Array<mqttMessage> = [];
	startMqtt(
		//"WS://wx.co6co.top:451/mqtt",
		mqqt_server,
		'/edge_app_controller_reply',
		(topic: any, message: any) => {
			const msg: mqttMessage = JSON.parse(message.toString());
			console.warn('收到信息', msg, typeof msg);
			arr.unshift(msg); //新增到数组起始位置
			noticeMessage.value = JSON.stringify(msg);
			console.warn(unique(arr));
		}
	);

	function unique(arr: Array<mqttMessage>) {
		const res = new Map();
		return arr.filter((a) => !res.has(a.UUID) && res.set(a.UUID, 1));
	}
	const OnPtz = (name: string, type: string) => {
		noticeMessage.value = '';
		let param = {
			payload: {
				BoardId: 'RJ-BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60',
				Event: '/app_network_query_v2',
			},
			qos: 0,
			retain: false,
		};
		Ref_Mqtt.value?.publish(
			'/edge_app_controller',
			JSON.stringify(param.payload)
		);
		console.warn(name, type);
	};
	//**end 打标签 */
</script>
<style lang="less">
	@import '../assets/css/tables.css';
	.el-header {
		height: 5vh;
	}
	#app .content .ui .el-container {
		height: 100%;
	}
	.el-main.ui {
		overflow: auto;
		height: 60vh;
		.el-header,
		.el-main {
			padding: 0;
			margin: 0;
			height: 27vh;
		}
		.el-header {
			height: 65vh;
		}
		::v-deep .Image {
			.el-col {
				height: 20rem;
			}
		}
		::v-deep .NavImage {
			height: 10rem;
		}
	}
	.el-footer {
		height: 30vh;
		position: relative;
		.ptzContent {
			width: 156px;
			height: 156px;
			position: absolute;
			top: 0;
			left: 0;
			right: 0;
			bottom: 0;
			margin: auto;
		}
	}

	.el-link {
		margin-right: 8px;
	}
	.el-link .el-icon--right.el-icon {
		vertical-align: text-bottom;
	}
	.handle-box {
		margin: 3px 0;
	}
	.handle-select {
		width: 120px;
	}

	.handle-input {
		width: 300px;
	}
	.table {
		width: 100%;
		font-size: 14px;
	}
	.red {
		color: #f56c6c;
	}
	.mr10 {
		margin-right: 10px;
	}
	.table-td-thumb {
		display: block;
		margin: auto;
		width: 40px;
		height: 40px;
	}
	.view .title {
		color: var(--el-text-color-regular);
		font-size: 18px;
		margin: 10px 0;
	}
	.view .value {
		color: var(--el-text-color-primary);
		font-size: 16px;
		margin: 10px 0;
	}

	::v-deep .view .radius {
		height: 40px;
		width: 70%;
		border: 1px solid var(--el-border-color);
		border-radius: 0;
		margin-top: 20px;
	}
	::v-deep .el-table tr,
	.el-table__row {
		cursor: pointer;
	}

	.formItem {
		display: flex;
		align-items: center;
		display: inline-block;
		.label {
			display: inline-block;
			color: #aaa;
			padding: 0 5px;
		}
	}

	::v-deep .el-dialog__body {
		height: 70%;
		overflow: auto;
	}
	.menuInfo {
		.el-menu {
			width: auto;
			.el-menu-item {
				padding: 10px;
				height: 40px;
			}
		}
	}

	/**
  
  ::v-deep .el-dialog{
      .el-dialog__header{padding: 5px;}
      .el-dialog__body{padding: 15px 5px;}
      .el-dialog__footer{padding: 5px;}
  } */
	.content {
		padding: 0;
	}
	.container {
		padding: 0;
	}
	.header {
		padding: 8px;
		font-size: 28px;
	}
	.collapse-btn {
		color: white;
	}
</style>
