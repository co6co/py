<template>
	<!--分屏-->
	<div class="playerList" id="playerList">
		<div class="video-list" :class="'video-split-' + playerList.splitNum">
			<template v-for="i in playerList.splitNum" :key="`video-item-${i}`">
				<div
					class="video-item splitNum"
					@click="onPlayerClick(i)"
					:class="{ active: i == playerList.currentWin }">
					<div class="player_container">
						{{ i }}
					</div>
					<div class="js">
						<stream-player
							ref="domRefs" 
							:stream="playerList.players[i - 1].url">
							 <!--:ref="(el: HTMLElement) => {setPlayerDom(i, el)}"-->
						</stream-player>
					</div>
				</div>
			</template>
		</div>
		<div class="video-tools">
			<ul>
				<li @click="onCloseAll()">
					<el-tooltip content="关闭所有">
						<el-icon>
							<CloseBold />
						</el-icon>
					</el-tooltip>
				</li>

				<li @click="onClose()">
					<el-tooltip content="关闭当前">
						<el-icon>
							<CircleClose />
						</el-icon>
					</el-tooltip>
				</li>

				<li @click="onScreenshot()">
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
							playerList.players[playerList.currentWin - 1].streamList.length >
							0
						">
						<el-select
							style="width: 90px"
							class="mr10"
							clearable
							v-model="playerList.players[playerList.currentWin - 1].url"
							placeholder="选择码流">
							<el-option
								v-for="(item, index) in playerList.players[ playerList.currentWin - 1 ].streamList"
								:key="index"
								:disabled="!item.valid"
								:label="item.name"
								:value="item.url" />
						</el-select>
					</div>
				</li>

				<li style="margin-left: auto" @click="onToggleFullScreens()">
					<el-tooltip content="全屏">
						<el-icon>
							<CanelFullScreen v-if="playerList.isFullScreen" />
							<FullScreen v-else />
						</el-icon>
					</el-tooltip>
				</li>
				<li @click="onSwitchSplitNum(4)">
					<el-tooltip content="4分屏">
						<el-icon> <FourScreen /> </el-icon>
					</el-tooltip>
				</li>

				<li @click="onSwitchSplitNum(2)">
					<el-tooltip content="2分屏">
						<el-icon> <TwoScreen /> </el-icon>
					</el-tooltip>
				</li>

				<li @click="onSwitchSplitNum(1)">
					<el-tooltip content="1分屏">
						<el-icon> <OneScreen /> </el-icon>
					</el-tooltip>
				</li>
			</ul>
		</div>
	</div>
</template>
<script lang="ts" setup>
	import {
		watchEffect,
		ref,
		watch,
		reactive,
		PropType,
		onMounted,
		onBeforeUnmount,
		computed,
	} from 'vue';
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
	import {
		OneScreen,
		TwoScreen,
		FourScreen,
		FullScreen,
		CanelFullScreen,
	} from '../../icons/screenIcon';
	import * as types from './types';
	import {types as dType} from '../../devices';
	import { toggleFullScreen } from '../../../utils';
	import {  streamPlayer } from '../../../components/stream';

	const props = defineProps({
		playerList: {
			type: Object as PropType<types.PlayerList>,
			required: true,
		},
	});  
	const   domRefs = ref<any>([]);
	const getCurrentDom=()=>{
		return domRefs.value[props.playerList.currentWin - 1]
	}
	const emit = defineEmits<{ (event: 'selected', index:number,data?:dType.DeviceData): void  }>();
 

	const onPlayerClick = (winIndex: number) => {
		props.playerList.currentWin = winIndex;
		props.playerList.currentStreams = props.playerList.players[props.playerList.currentWin - 1].streamList;
		emit("selected",winIndex,props.playerList.players[winIndex - 1].data)
	};

	const onToggleFullScreens = () => {
		//全屏/退出全屏
		var ele = document.getElementById('playerList');
		if (ele) toggleFullScreen(ele, !props.playerList.isFullScreen);
		props.playerList.isFullScreen = !props.playerList.isFullScreen;
	};

	const onSwitchSplitNum = (n: number) => {
		props.playerList.splitNum = n;
	};

	const onCloseAll = () => {
		for (let i = 0; i < props.playerList.players.length; i++) {
			const dom =domRefs.value[i]
			if (dom.stop) dom.stop();
		}
	};
	const onClose = () => { 
		const dom = getCurrentDom(); 
		if (dom.stop)dom.stop();
	};
	const onScreenshot=()=>{
		const dom = getCurrentDom(); 
		if (dom.jess_player&& dom.jess_player.screenshot) dom.jess_player.screenshot()
		else console.warn("player.screenshot is undenfine")
	}
</script>

<style scoped lang="less">
	@import '../../../assets/css/player-split.css';
	 
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
</style>
