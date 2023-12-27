<template>
	<div class="container-layout">
		<el-container>
			<el-aside width="200px">
				<el-scrollbar>
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
								:filter-node-method="tree_module.filterNode" />
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
				</el-scrollbar>
			</el-aside>
			<el-main>
				<el-scrollbar>
					<div class="content">
						<el-row>
							<el-col :span="19" style="height: 100%">
								<div class="playerList" id="playerList">
									<div
										class="video-list"
										:class="'video-split-' + playerList.splitNum">
										<template v-for="i in playerList.splitNum">
											<div
												class="video-item splitNum"
												@click="onPlayerClick(i)"
												:class="{ active: i == playerList.currentWin }">
												<div class="player_container">
													{{ i }}
												</div>
												<div class="js">
													<stream-player 
													:stream="
														playerList.players[i - 1].url
													"></stream-player>
												</div>
												
											</div>
										</template>
									</div>
									<div class="video-tools">
										<ul>
											<li>
												<el-tooltip content="关闭所有">
													<el-icon>
														<CloseBold />
													</el-icon>
												</el-tooltip>
											</li>

											<li>
												<el-tooltip content="关闭当前">
													<el-icon>
														<CircleClose />
													</el-icon>
												</el-tooltip>
											</li>

											<li>
												<el-tooltip content="截图">
													<el-icon>
														<PictureFilled />
													</el-icon>
												</el-tooltip>
											</li>

											<li>
												<span class="form-label" id="streamFullNameId"></span>
											</li>

											<li>
												<div
													class="select-form-item"
													v-show="
														playerList.players[playerList.currentWin - 1]
															.streamList.length > 0
													">
													<el-select
														style="width: 160px"
														class="mr10"
														clearable
														v-model="
															playerList.players[playerList.currentWin - 1].url
														"
														placeholder="选择码流">
														<el-option
															v-for="(item, index) in playerList.players[
																playerList.currentWin - 1
															].streamList"
															:key="index"
															:label="item.name"
															:value="item.url" />
													</el-select>
												</div>
											</li>

											<li style="margin-left: auto" @click="onFullScreens">
												<el-tooltip content="全屏">
													<el-icon>
														<FullScreen />
													</el-icon>
												</el-tooltip>
											</li>
											<li @click="onSwitchSplitNum(4)">
												<el-tooltip content="4">
													<el-icon>
														<FullScreen />
													</el-icon>
												</el-tooltip>
											</li>

											<li @click="onSwitchSplitNum(2)">
												<el-tooltip content="2">
													<el-icon>
														<FullScreen />
													</el-icon>
												</el-tooltip>
											</li>

											<li @click="onSwitchSplitNum(1)">
												<el-tooltip content="1">
													<el-icon>
														<FullScreen>1</FullScreen>
													</el-icon>
												</el-tooltip>
											</li>
										</ul>
									</div>
								</div>
							</el-col>
							<el-col :span="5">
								<div class="content">
									<ptz @ptz="OnPtz"></ptz>
								</div>
							</el-col>
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
		CloseBold,
	} from '@element-plus/icons-vue';
	import * as api from '../api/device';
	import { stream, ptz, streamPlayer } from '../components/stream';
	import * as p from '../components/stream/src/types/ptz';
	import { toggleFullScreen } from '../utils';
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
	interface player {
		url: string;
		streamList: Array<stream_source>;
	}
	interface PlayerList {
		splitNum: number;
		currentWin: number;
		currentStreams: Array<stream_source>;
		players: Array<player>;
	}
	const playerList = reactive<PlayerList>({
		splitNum: 4,
		currentWin: 1,
		currentStreams: [],
		players: [
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
		],
	});

	const onPlayerClick = (winIndex: number) => {
		playerList.currentWin = winIndex;
		playerList.currentStreams =
			playerList.players[playerList.currentWin - 1].streamList;
	};

	const onFullScreens = () => {
		//全屏
		var ele = document.getElementById('playerList');
		if (ele) {
			toggleFullScreen(ele);
		}
	};

	const onSwitchSplitNum = (n: number) => {
		playerList.splitNum = n;
	};

	const onNodeCheck = (row?: any) => {
		tree_module.currentItem = row;
		if (row.streams && typeof row.streams == 'string') {
			let stream = JSON.parse(row.streams);
			playerList.players[playerList.currentWin - 1].streamList = stream;
			playerList.players[playerList.currentWin - 1].url = stream[0].url;
		} else {
			playerList.players[playerList.currentWin - 1].url = '';
			playerList.players[playerList.currentWin - 1].streamList = [];
			ElMessage.warning('未配置设备流地址');
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
<style scoped lang="less">
	@import '../assets/css/player-split.css';
	@color: #000; // rgba(255, 255, 255, 0.2);
	@bgcolor: #fff; // #464444;
	.el-container {
		height: 82vh;
		overflow: hidden;
		.el-scrollbar {
			background: @bgcolor;
		}

		.el-aside {
			::v-deep .el-card__body {
				height: 59vh;
			} 
		}
		.el-main {
			padding: 0;
			margin: 0 15px;
			::v-deep .el-scrollbar__view {
				height: 100%;
			}
			.el-row {
				height: 100%;
			}
			.el-scrollbar {
				.playerList {
					.el-icon {
						cursor: pointer;
						font-size: 23px;
						vertical-align: middle;
						padding-bottom: 7px;
						&:hover {
							color: red;
						}
					}
					.video-item {
						position: relative;
						.js {
							position: absolute;
							width: calc(100% - 4px);
							height: calc(100% - 4px); 
							left: 2px;
							top: 2px;
						}
					}
				}
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
