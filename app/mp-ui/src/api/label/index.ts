import { json } from 'stream/consumers';
import request  from '../../utils/request';
const BASE_URL="/api/biz/label"
const mark_BASE_URL="/api/biz/marklabel"


export const get_exist_svc = (id:number,name:string): Promise<IPageResponse> => {
    return request.get(`${BASE_URL}/exist/${id}/${name}`);
}; 
//获取状态列表
export const get_list_svc = (data:any): Promise<IPageResponse> => {
    return request.post(`${BASE_URL}`,data);
};  
export const add_svc = (data:any): Promise<IResponse> => {
    return request.put(`${BASE_URL}`,data);
}; 
export const edit_svc = (id:number,data:any): Promise<IResponse> => {
    return request.post(`${BASE_URL}/${id}`, data);
}; 
export const del_svc = (id:number): Promise<IResponse> => {
    return request.delete(`${BASE_URL}/${id}`, {});
};  

//获取可打的标签类别
export const mark_list_svc = ( ): Promise<IPageResponse> => {
    return request.get(`${mark_BASE_URL}`);
};  
export const marked_list_svc = (processId:number ): Promise<IPageResponse> => {
    return request.get(`${mark_BASE_URL}/${processId}`);
};  
export const mark_label_svc = (processId:number, data:{ appendIds:Array<Number> ,removeIds:Array<Number>}): Promise<IPageResponse> => {
    return request.post(`${mark_BASE_URL}/${processId}`,data);
};  

