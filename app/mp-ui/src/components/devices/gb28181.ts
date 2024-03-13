import { onUnmounted, onMounted, defineComponent } from 'vue';
import * as gb_api from '../../api/deviceState';

const Interval = 30000; //30s
let deviceState_timer: NodeJS.Timeout | null = null;
let isRunning = false;
//通道在线
export const GbDeviceState = (bck: (data: gb_api.gbDeviceState[]) => void) => {
	if (deviceState_timer) clearInterval(deviceState_timer);
	if (isRunning) return;
	isRunning = true;
	gb_api
		.get_gb_device_state()
		.then((res) => {
			let stateArray = res.data.data;
			if (bck) bck(stateArray);
		})
		.finally(() => {
			deviceState_timer = setInterval(GbDeviceState, Interval, bck);
			isRunning = false;
		});
};
//rtc 设备 在线状态
let rtc_timer: NodeJS.Timeout | null = null;
let rtc_isRunning = false;
export const RtcOnlineState = (bck: (data: gb_api.gbTaklerOnlineList) => void) => {
	if (rtc_timer) clearInterval(rtc_timer);
	if (rtc_isRunning) return;
	rtc_isRunning = true;
	gb_api
		.get_takler_online()
		.then((res) => { 
			if (bck) bck(res.data);
		})
		.finally(() => {
			rtc_timer = setInterval(RtcOnlineState, Interval, bck);
			rtc_isRunning = false;
		});
};

//rtc 设备  空闲状态
let rtc_session_timer: NodeJS.Timeout | null = null;
let rtc_session_isRunning = false;
export const RtcSessionState = (
	bck: (data: gb_api.gbTaklerOnlineSessionList) => void
) => {
	if (rtc_session_timer) clearInterval(rtc_session_timer);
	if (rtc_session_isRunning) return;
	rtc_session_isRunning = true;
	gb_api
		.get_takler_online_session()
		.then((res) => { 
			if (bck) bck(res.data);
		})
		.finally(() => {
			rtc_session_timer = setInterval(RtcSessionState, Interval, bck);
			rtc_session_isRunning = false;
		});
};
onMounted(() => {});
onUnmounted(() => {
	if (deviceState_timer)
		clearInterval(deviceState_timer), (deviceState_timer = null);
});
