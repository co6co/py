//派单相关API 
import request from '../../utils/request'; 


import {type Ipd_response} from './types'
const Beat_URL = '/user_heartbeat'; //心跳
let orders_URL = '/assign_orders'; //派单
const audit_URL = '/manual_audit_result'; //派单
const related_boat_changed_url='/related_boat' ;// 关联船后需要需要通知分配系统
const boat_priority_changed_URL='/boat_priority' ;// 船优先级改变需要通知分配系统
const rule_priority_changed_URL="/rule_priority";//  规则优先级改变需要通知分配系统
const auto_push_boat_changed_URL="/auto_push_change_boat"; //仅需AI审核的船发生改变 通知分配系统
const manual_review_changed_URL= "/manual_review_change";//	规则人工复审 状态发生改变 通知分配系统



const pd_base_url=import.meta.env.VITE_PD_BASEURL
console.info("..dp..",pd_base_url)

const debug = Boolean(Number(import.meta.env.VITE_IS_DEBUG));
if (debug)orders_URL= import.meta.env.VITE_UI_PATH+ orders_URL+".json" 
export interface IheartBeat_param {
	user_id: number;
	time: string;
}
//派单参数
export interface Iorders_param {
	user_id: number;
	record_num: number;
}
export interface Iorders_record {
	id: number;
	web_record_data: string;
	web_record_id: number; 
	dev_record_time: string;
	web_record_time: string;
	audit_download_time: string;
	audit_infer_time: string;
	audit_auto_push_time: string;
	manual_audit_time: string;
	manual_push_time: string;
	boat_id: number;
	boat_name: string;
	vcam_serial: string;
	device_serial: string;
	boat_pos: string;
	break_rules: string;
	vio_name: string;
	flow_status: number;
	re_download_num: number;
	video_save_path: string;
	pic1_save_path: string;
	anno_video_save_path: string;
	anno_pic1_save_path: string;
	manual_audit_status: string;
	manual_audit_result?: number;
	label?:string;
	program_audit_result: number;
	manual_audit_remark: string;
	audit_log_path: string;
	
	audit_id: string;
	auditor_id?:number
	
	job_serial: string;
	create_time: string;
	create_user: number;
	update_time: string;
	update_user: string;  
}
//派单响应
export interface IOrders_response extends Ipd_response {
	job_id: bigint; //# 订单任务流水号
	records:   Iorders_record[]; 
	time_out:number //单位分钟
}

//单条审核记录
export interface Iaudit_record{
	job_id:BigInt;				// 63754308716557023406, # 订单任务流水号
    id:number;					// 7211, # 记录id
    manual_audit_result:number; // 1, # 审核结果 
    auditor_id:number;			// 3 # 审核员id
	label?:string;				//人工标签
}
//单条审核记录
export interface Iaudit_param{
    user_id: number;//3, # 用户id
    records:  Iaudit_record[];
}

export const isSuccess=(res?:Ipd_response)=>{
	if (res&& res.state&&res.state=="200") return true
	if(res && !res.message)res.message=`${res.state}:请求出错`
	return false
}

/**
 * //发送心跳
 * @param beat  
 * @returns 
 */

export const heartbeat = (beat: IheartBeat_param): Promise<Ipd_response> => { 
	return request.post(`${Beat_URL}`,beat,{baseURL:pd_base_url});
};
/**
 * //请求派单
 * @param param  
 * @returns 
 */

export const assign_orders=(param:Iorders_param):Promise<IOrders_response>=>{
    return request.post(`${orders_URL}`, param,{baseURL:pd_base_url,timeout:30000});
}
/**
 * //人工审核
 * @param param 
 * @returns 
 */

export const manual_audit=(param:Iaudit_param):Promise<Ipd_response>=>{ 
    return request.post(`${audit_URL}`, param,{baseURL:pd_base_url});
}

/**
 * 用户分配的船 改变
 * 需通知 分配系统
 * @param user_id 
 * @returns 
 */
export const related_boat_changed_svc=(user_id:number):Promise<Ipd_response>=>{ 
    return request.post(`${related_boat_changed_url}`, {user_id:user_id},{baseURL:pd_base_url});
}

/**
 * 船优先级 改变
 * 需要通知分配系统
 * @param user_id 
 * @returns 
 */
export const boat_priority_changed_svc=(user_id:number):Promise<Ipd_response>=>{ 
    return request.post(`${boat_priority_changed_URL}`, {user_id:user_id},{baseURL:pd_base_url});
}
/**
 * 规则优先级 改变
 * 通知 分配系统
 * @param user_id 
 * @returns 
 */ 
export const rule_priority_changed_svc=(user_id:number):Promise<Ipd_response>=>{ 
    return request.post(`${rule_priority_changed_URL}`, {user_id:user_id},{baseURL:pd_base_url});
}

/**
 * 自动推送检测结果 船 改变 
 * 
 * 通知 分配系统
 * @param user_id 
 * @returns 
 */ 
export const auto_audit_boat_changed_svc=(user_id:number):Promise<Ipd_response>=>{ 
    return request.post(`${auto_push_boat_changed_URL}`, {user_id:user_id},{baseURL:pd_base_url});
}
 /**
 * 规则人工复审状态变更
 * 通知 分配系统
 * @param user_id 
 * @returns 
 */ 
export const manual_review_state_changed_svc=(user_id:number):Promise<Ipd_response>=>{ 
    return request.post(`${manual_review_changed_URL}`, {user_id:user_id},{baseURL:pd_base_url});
} 