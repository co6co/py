<template>
	<div class="container-layout">
		<el-container>
			<el-main>
				<el-scrollbar>
					<div class="content">
						<biz-player :player-list="playerList"></biz-player>
					</div>
				</el-scrollbar>
			</el-main>
			<el-footer>
				<el-row>
					<el-col :span="12">
						<el-card class="box-card">
							<!--header-->
							<template #header>
								<div class="card-header">
									<el-input
										v-model="tree_module.query.name"
										placeholder="点位名称">
										<template #append>
											<el-button :icon="Search" @click="tree_module.onSearch" />
										</template>
									</el-input>
								</div>
							</template>
							<!--content-->
							<div class="content">
								<el-tree
									v-if="hasData"
									highlight-current
									@node-click="onNodeCheck"
									ref="tree"
									class="filter-tree"
									:data="tree_module.data"
									:props="tree_module.defaultProps"
									default-expand-all
									:filter-node-method="tree_module.filterNode">
									<template #default="{ node, data }">
										<span>
											<!--
										<i v-if="node.expanded" > 
											<el-icon><ArrowUp /></el-icon>
										</i>
										 
										<i v-else>
											<el-icon><ArrowDown /></el-icon>
										</i>
										-->
											<!-- 没有子级所展示的图标 -->
											<i v-if="!data.devices"
												><el-icon><VideoCamera /></el-icon
											></i>
											<i v-else-if="data.devices"
												><el-icon><Avatar /></el-icon
											></i>

											{{ node.label }}
										</span>
									</template>
								</el-tree>

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
										@current-change="tree_module.pageChange" /></div
							></template>
						</el-card>
					</el-col>
					<el-col :span="12">
						<div class="content">
							<talker
								ref="talkerRef"
								:talk-no="tree_module.currentDevice?.talkbackNo"></talker>
							<ptz @ptz="OnPtz"></ptz>
						</div>
					</el-col>
				</el-row>
			</el-footer>
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
		CloseBold,
		VideoCamera,
		Avatar,
		ArrowUp,
		ArrowDown,
	} from '@element-plus/icons-vue';
	
	//import * as api from '../api/device';
	import * as api from '../api/site';
	import { stream, ptz, streamPlayer } from '../components/stream';
	import * as p from '../components/stream/src/types/ptz';
	import { toggleFullScreen } from '../utils';
	import { useMqtt, mqqt_server } from '../utils/mqtting';
	import * as d from '../store/types/devices';
	import { showLoading, closeLoading } from '../components/Logining';
	import { talker } from '../components/devices';
	import { bizPlayer, types } from '../components/biz';

	const deviceName = ref('');
	const tree = ref(null);
	interface Tree {
		[key: string]: any;
	}
	interface Query extends IpageParam {
		name: string;
	}
	interface dataItem {}
	interface deviceItem {
		streams: string;
		sip: string;
		talkbackNo: number;
		channel1_sip: string;
		channel2_sip: string;
	}
	interface tree_module {
		query: Query;
		data: Array<dataItem>;
		currentItem?: dataItem;
		currentDevice?: deviceItem;
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
			pageSize: 20,
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
			children: 'devices',
			label: 'name',
		},
	});
	// 获取表格数据
	const getData = () => {
		showLoading();
		api
			.list_svc(tree_module.query)
			.then((res) => {
				if (res.code == 0) {
					for (let i = 0; i < res.data.length; i++) {
						//如果 devices 只有1条，移动值 为 res.data[i] 属性
						if (res.data[i].devices && res.data[i].devices.length == 1) {
							res.data[i].device = res.data[i].devices[0];
							delete res.data[i].devices;
						}
					}
					tree_module.data = res.data;
					tree_module.total = res.total || -1;
				} else {
					ElMessage.error(res.message);
				}
			})
			.finally(() => {
				closeLoading();
			});
	};
	const hasData = computed(() => tree_module.data.length > 0);
	getData();
	/** 播放器 */

	const playerList = reactive<types.PlayerList>({
		splitNum: 1,
		isFullScreen: false,
		currentWin: 1, 
		currentStreams: [],
		players: [
			{  url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
		],
	});
	const play = (streams: String | { url: string; name: string }) => {
		let streamArr = null;
		if (streams && typeof streams == 'string') streamArr = JSON.parse(streams);
		else streamArr = streams;
		if (streamArr) {
			playerList.players[playerList.currentWin - 1].streamList = streamArr;
			playerList.players[playerList.currentWin - 1].url = streamArr[0].url;
		} else {
			playerList.players[playerList.currentWin - 1].url = '';
			playerList.players[playerList.currentWin - 1].streamList = [];
			ElMessage.warning('未配置设备流地址');
		}
	};
	const onNodeCheck = (row?: any) => {
		tree_module.currentItem = row;
		//只有一个设备的点
		if (row.device) {
			let device = row.device;
			tree_module.currentDevice = device;
			let stream = device.streams;
			play(stream);
		}
		//有多个设备的点 ，仅展开
		else if (row.devices) console.info('展开');
		else {
			//点位
			tree_module.currentDevice = row;
			play(row.streams);
		}
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

	const talkerRef = ref();

	const OnPtz = (name: p.ptz_name, type: p.ptz_type) => {
		// 对接
		if (type == 'starting' && name == 'center') {
			talkerRef.value.connect();
		} else if (type == 'stop' && name == 'center') {
			talkerRef.value.disconnect();
		} else {
			let param = {
				payload: {
					BoardId: 'RJ-BOX3-733E5155B1FBB3C3BB9EFC86EDDACA60',
					Event: '/app_network_query_v2',
				},
				qos: 0,
				retain: false,
			};
			let ptzcmd = 'A50F010800FA00B7';
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
			let sip = tree_module.currentDevice?.sip;
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
		}
	};
	//**end 打标签 */
</script>
<style scoped lang="less">
	@import '../assets/css/player-split.css';
	@color: #000; // rgba(255, 255, 255, 0.2);
	@bgcolor: #fff; // #464444;
	#app .content .el-container {
		height: 100vh;
		.el-scrollbar {
			background: @bgcolor;
		}
		.el-footer {
			height: 40vh;
		}
		.el-main {
			height: 60vh;
			padding: 0;
			margin: 0 15px;
			::v-deep .el-scrollbar__view {
				height: 100%;
			}
			.el-scrollbar {
				.content {
					padding: 10px;
					.box {
						width: 50%;
						display: inline-block;
						margin-right: 5px;
					}
				}
			}
		}
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
