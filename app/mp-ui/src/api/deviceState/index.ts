 
import request from '../../utils/request';
const baseURL = 'https://stream.jshwx.com.cn:8441';

const GB_URL = `${baseURL}/list?dev=gb`;
const RTC_URL = `${baseURL}/list?atv=rtc-device`; //对讲设备是否在线
const RTC_SESSION_URL = `${baseURL}//list?atv=rtc-session`; //对讲设备是否空闲


const gb_BASE_URL="/api/xss/list"  

export interface gbDeviceState {
	uri: string;
	parent: string;
	contact: string;
	port: number;
	mode: string;
	name: string;
	factory: string;
	model: string;
}

export interface gbTaklerDeviceMedia{
	device:number
	track:number
	gburl:string
}
export interface gbTaklerDevice{
	"peer-id":string
	"device-id":number,
	ipaddr:string,
	port:number,
	media:Array<gbTaklerDeviceMedia>
} 
//语音对讲设备在线状态
export interface gbTaklerOnlineList{
	count:number
	data:Array<gbTaklerDevice> 
}

export interface CallInfo{
	from_userid:string //unknown
	media:string	//talk
	device:number	//6
	track:number //-1
	gburl:string; //none
}
export interface gbTaklerBusy{
	"session-id":string,
	"caller-id":string,
	"callee-id":string,
	"call-info":CallInfo,
	"duration":number
}
//语音对讲设备忙
export interface gbTaklerOnlineSessionList{
	count:number
	data:Array<gbTaklerBusy> 
} 
export const get_gb_device_state = (): Promise<{
	data: Array<gbDeviceState>;
}> => {
    return request.get(`${gb_BASE_URL}?dev=gb` );
};

export const get_takler_online = (): Promise<gbTaklerOnlineList> => {
    return request.get(`${gb_BASE_URL}?atv=rtc-device` );
};

export const get_takler_online_session = (): Promise<gbTaklerOnlineSessionList> => {
	return request.get(`${gb_BASE_URL}?atv=rtc-session` );
};
