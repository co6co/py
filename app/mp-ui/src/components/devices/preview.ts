import { reactive,ref } from 'vue';
import { types } from '../../components/biz';
import * as dType from './src/types';
import { ElMessage } from 'element-plus';

export const playerList = reactive<types.PlayerList>({
	splitNum: 1,
	isFullScreen: false,
	currentWin: 1,
	currentStreams: [],
	players: [
		{ url: '', streamList: [{ name: '', url: '', valid: false }] },
		{ url: '', streamList: [{ name: '', url: '', valid: false }] },
		{ url: '', streamList: [{ name: '', url: '', valid: false }] },
		{ url: '', streamList: [{ name: '', url: '', valid: false }] },
	],
}); 

export const currentItem =reactive<dType.CurrentItem>( {});
export const onClickNavDevice = (
	box?:dType.BoxDevice,	
	device?: dType.DeviceData,
	streams?: dType.Stream[]
) => {
	currentItem.box= box;
	currentItem.device=device
	//todo debug reactive 复制给一个 reactive 的值 会不会出现异常
	playerList.players[playerList.currentWin - 1].data = currentItem ; 
	if (streams) {
		playerList.players[playerList.currentWin - 1].streamList = streams;
		if (streams[0].valid) playerList.players[playerList.currentWin - 1].url = streams[0].url; 
		else playerList.players[playerList.currentWin - 1].url='', ElMessage.warning('通道1,未连接,请选择其他通道！');
	} else {
		playerList.players[playerList.currentWin - 1].url = '';
		playerList.players[playerList.currentWin - 1].streamList = [];
		ElMessage.warning('未配置设备流地址');
	}  
}; 
export const onPlayerChecked = (index: number, item?:dType.CurrentItem) => {
	if (item){
		currentItem.box=item.box
		currentItem.device=item.device
	}else{
		currentItem.box=undefined
		currentItem.device=undefined
	} 
	
};
