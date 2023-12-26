<template>
	<div class="container-layout">
		<el-container>
			<el-aside width="200px">
				<el-card class="box-card">
					<!--header-->
					<template #header>
						<div class="card-header">
							<el-input v-model="tree_module.query.name" placeholder="点位名称">
								<template #append>
									<el-button :icon="Search" @click="tree_module.onSearch" />
								</template>
							</el-input>
						</div>
					</template>
					<!--content-->
					<div>
						<el-tree
							v-if="hasData"
							highlight-current
							@node-click="onNodeCheck"
							ref="tree"
							class="filter-tree"
							:data="tree_module.data"
							:props="tree_module.defaultProps"
							default-expand-all
							:filter-node-method="tree_module.filterNode"
						/>
						<el-empty v-else></el-empty>
					</div>
					<!--footer-->
					<template #footer>
						<div class="context">
							<el-pagination
								v-if="hasData"
								background
								layout="prev,next"
								:total="tree_module.total"
								:current-page="tree_module.query.pageIndex"
								:page-size="tree_module.query.pageSize"
								@current-change="tree_module.pageChange"
							/></div
					></template>
				</el-card>
			</el-aside>
			<el-main>
				<el-scrollbar>
					<div style="padding: 5px">
						<el-row :gutter="24">
							<el-col :span="20"
								><stream :sources="player.sources"></stream
							></el-col>
							<el-col :span="4"><ptz @ptz="OnPtz"></ptz></el-col>
						</el-row>
					</div>
				</el-scrollbar>
			</el-main>
		</el-container>
	</div>
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
		ElMessage,
		ElMessageBox,
		FormRules,
		FormInstance,
		ElTreeSelect,
	} from 'element-plus';
	import { TreeNode } from 'element-plus/es/components/tree-v2/src/types';
	import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type';
	import {
		Delete,
		Edit,
		Search,
		Compass,
		MoreFilled,
		Download,
	} from '@element-plus/icons-vue';
	import * as api from '../api/device';
	import { stream, ptz } from '../components/stream';
	import * as p from '../components/stream/src/types/ptz';

	import { useMqtt, mqqt_server } from '../utils/mqtting';
	import * as d from '../store/types/devices';

	const deviceName = ref('');
	const tree = ref(null);
	interface Tree {
		[key: string]: any;
	}
	interface Query extends IpageParam {
		name: string;
	}
	interface dataItem {}
	interface tree_module {
		query: Query;
		data: Array<dataItem>;
		currentItem?: dataItem;
		total: number;
		defaultProps: { children: String; label: String };
		filterNode: (value: string, data: Tree) => boolean;
		pageChange: (val: number) => void;
		onSearch: () => void;
	}
	const tree_module = reactive<tree_module>({
		query: {
			name: '',
			pageIndex: 1,
			pageSize: 10,
			order: 'asc',
			orderBy: '',
		},
		data: [],
		total: 0,
		filterNode: (value: string, data: Tree) => {
			if (!value) return true;
			return data.label.includes(value);
		},
		// 分页导航
		pageChange: (val: number) => {
			tree_module.query.pageIndex = val;
			getData();
		},
		onSearch: () => {
			getData();
		},
		defaultProps: {
			children: 'children',
			label: 'name',
		},
	});
	// 获取表格数据
	const getData = () => {
		api.list_svc(tree_module.query).then((res) => {
			if (res.code == 0) {
				tree_module.data = res.data;
				tree_module.total = res.total || -1;
			} else {
				ElMessage.error(res.message);
			}
		});
	};
	const hasData = computed(() => tree_module.data.length > 0);
	getData();
	/** 播放器 */
	interface player_sources {
		sources: Array<stream_source>;
	}
	const player = reactive<player_sources>({ sources: [] });

	const onNodeCheck = (row?: any) => {
		tree_module.currentItem = row;
		console.info(row)
		if (row.streams && typeof row.streams == 'string') player.sources = JSON.parse(row.streams);
		else player.sources = [],ElMessage.warning("未配置设备流地址");
		/**
	  [

		  {url:`http://wx.co6co.top:452/flv/vlive/${item.ip}.flv`,name:"HTTP-FLV"},
			{url:`ws://wx.co6co.top:452/ws-flv/vlive/${item.ip}.flv`,name:"WS-FLV"},
			{url:`webrtc://wx.co6co.top:452/rtc/vlive/${item.ip}`,name:"webrtc"},
			{url:`http://wx.co6co.top:452/vhls/${item.ip}/${item.ip}_live.m3u8`,name:"HLS(m3u8)"}
		]   */
	};
	/** ptz */
	const { startMqtt, Ref_Mqtt } = useMqtt();
	interface mqttMessage {
		UUID?: string;
	}
	let arr: Array<mqttMessage> = [];
	startMqtt(
		//'WS://wx.co6co.top:451/mqtt',
		mqqt_server,
		//'WS://yx.co6co.top/mqtt',
		'/edge_app_controller_reply',
		(topic: any, message: any) => {
			const msg: mqttMessage = JSON.parse(message.toString());
			arr.unshift(msg); //新增到数组起始位置
			let data = unique(arr);
			alert(JSON.stringify(data));
			console.warn(unique(arr));
		}
	);

	function unique(arr: Array<mqttMessage>) {
		const res = new Map();
		return arr.filter((a) => !res.has(a.UUID) && res.set(a.UUID, 1));
	}

	const mediaStreamStarting = ref(false);
	const OnPtz = (name: p.ptz_name, type: p.ptz_type) => {
		// 对接
		if (type == 'starting' && name == 'center') {
			mediaStreamStarting.value = true;
			navigator.mediaDevices
				.getUserMedia({
					audio: true /*, video:{width:1280, height:720,facingMode: "user" }*/,
				})
				.then(function (mediaStream) {
					if (!mediaStreamStarting.value) {
						mediaStream.getTracks()[0].stop();
					}
				})
				.catch(function (error) {});
		} else if (type == 'stop' && name == 'center') {
			mediaStreamStarting.value = false;
		} else {
			console.warn(name, type);
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
		}
	};
	//**end 打标签 */
</script>
<style lang="less">
	@import '../assets/css/tables.css';
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
	.el-card__header {
		padding: 1px;
	}
	.el-card__body {
		padding: 5px;
		min-height: 289px;
	}
	.el-card__footer {
		.context {
			text-align: center;
		}
		padding: 5px;
	}

	.el-tree--highlight-current
		.el-tree-node.is-current
		> .el-tree-node__content {
		background: #bdbdc5;
		color: #f56c6c;
	}
</style>
