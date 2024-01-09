<template>
	<div class="container-layout">
		<el-container>
			<el-aside width="200px">
				<device-nav @node-click="play"></device-nav>
			</el-aside>
			<el-main>
				<el-scrollbar>
					<div class="content">
						<el-row>
							<el-col :span="19" style="height: 100%">
								<biz-player :player-list="playerList"></biz-player>
							</el-col>
							<el-col :span="5">
								<div class="content">
									<talker
										ref="talkerRef"
										:talk-no="currentDeviceData.data?.talkbackNo"></talker>
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
	import { talker ,deviceNav,types as dType} from '../components/devices';
	import { bizPlayer, types } from '../components/biz'; 

	const deviceName = ref('');
	
	/** 播放器 */ 
	const playerList = reactive<types. PlayerList>({
		splitNum: 1,
		isFullScreen: false,
		currentWin: 1,
		currentStreams: [],
		players: [
			{ dom: {}, url: '', streamList: [{ name: '', url: '' }] },
			{ dom: {}, url: '', streamList: [{ name: '', url: '' }] },
			{ dom: {}, url: '', streamList: [{ name: '', url: '' }] },
			{ dom: {}, url: '', streamList: [{ name: '', url: '' }] },
		],
	}); 
	const currentDeviceData=reactive<{data?:dType.deviceItem}>({})
	const play = (streams: String | { url: string; name: string },device:dType.deviceItem) => {
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
		currentDeviceData.data=device
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
			let sip = currentDeviceData.data?.sip;
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
	.el-container {
		height: 82vh;
		overflow: hidden;
		.el-scrollbar {
			background: @bgcolor;
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
