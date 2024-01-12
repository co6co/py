<template>
	<div class="container-layout">
		<el-container>
			<el-main>
				<el-scrollbar>
					<div class="content">
						<biz-player :player-list="playerList" @selected="onPlayerChecked"></biz-player>
					</div>
				</el-scrollbar>
			</el-main>
			<el-footer>
				<el-row>
					<el-col :span="12">
						<device-nav @node-click="onClickNavDevice"></device-nav>
					</el-col>
					<el-col :span="12">
						<div class="content">
							<takler-ptz :current-device-data="currentDeviceData.data"></takler-ptz> 
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
	import * as api from '../api/site';
	import { stream, ptz, streamPlayer } from '../components/stream';
	import * as p from '../components/stream/src/types/ptz';
	import { toggleFullScreen } from '../utils';
	import { useMqtt, mqqt_server } from '../utils/mqtting';
	import * as d from '../store/types/devices';
	import { showLoading, closeLoading } from '../components/Logining';
	import { talker, deviceNav, types as dType } from '../components/devices';
	import { bizPlayer,taklerPtz, types } from '../components/biz';

 

	/** 播放器 */
	const playerList = reactive<types.PlayerList>({
		splitNum: 1,
		isFullScreen: false,
		currentWin: 1,
		currentStreams: [],
		players: [
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
			{ url: '', streamList: [{ name: '', url: '' }] },
		],
	});
	const currentDeviceData = reactive<{ data?: dType.deviceItem }>({});
	const onClickNavDevice = (
		streams: String | { url: string; name: string },
		device: dType.deviceItem
	) => {
		let streamArr = null;
		if (streams && typeof streams == 'string') streamArr = JSON.parse(streams);
		else streamArr = streams;
		playerList.players[playerList.currentWin - 1].data=device
		if (streamArr) {
			playerList.players[playerList.currentWin - 1].streamList = streamArr;
			playerList.players[playerList.currentWin - 1].url = streamArr[0].url;

		} else {
			playerList.players[playerList.currentWin - 1].url = '';
			playerList.players[playerList.currentWin - 1].streamList = [];
			ElMessage.warning('未配置设备流地址');
		}
		currentDeviceData.data = device;
	}; 

	const onPlayerChecked=(index:number,data?: dType.deviceItem)=>{ 
		currentDeviceData.data = data; 
	}
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
			height: 48vh;
			padding: 0;
			.content {
				padding: 0 10px;
			}
			::v-deep .el-card__body {
				height: 22vh;
			}
		}
		.el-main {
			height: 52vh;
			padding: 0;
			margin: 0;
			::v-deep .el-scrollbar__view {
				height: 100%;
			}
			.el-scrollbar {
				.content {
					padding: 0;
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
