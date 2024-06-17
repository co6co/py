import request from '../../utils/request';  
const pd_base_url=import.meta.env.VITE_PD_BASEURL
import {type Ipd_response} from './types'
const get_URL = '/get_query_rule'; 
const set_URL = '/set_query_rule'; 

export type PriorityLevel =1|2|3|4
export type PriorityValue =0|1|2|3|4
export interface IPriority{
    ai_pass_priority:PriorityValue
    boat_priority: PriorityValue
    latest_time_priority: PriorityValue
    rule_priority:PriorityValue
}
export interface IRecode{
    user_id:number
    ai_audited_no_review: number
    expiration_time: number
    priority:IPriority
} 
export interface IAuditConfig  {
    data:IRecode 
} 
export interface IAuditConfigResponse extends Ipd_response{
    data:IRecode 
} 

//查询配置
export const get_svc = (userId:number): Promise<IAuditConfigResponse> => { 
	return request.post(`${get_URL}`, {user_id:userId},{baseURL:pd_base_url});
}; 
//设置配置
export const set_svc=(param:IAuditConfig):Promise<Ipd_response>=>{ 
    return request.post(`${set_URL}`, param,{baseURL:pd_base_url});
}