 
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
 
export const get_gb_device_state = (): Promise<{
	data: Array<gbDeviceState>;
}> => {
    return request.get(`${gb_BASE_URL}?dev=gb` );
};

export const get_rtc_device_state = () => {
    return request.get(`${gb_BASE_URL}?atv=rtc-device` );
};

export const get_rtc_session_state = () => {
	return request.get(`${gb_BASE_URL}?atv=rtc-session` );
};
