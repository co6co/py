import request  from '../../utils/request';
const BASE_URL="/api/biz/task"

//获取状态列表
export const get_status_svc = (): Promise<IPageResponse> => {
    return request.get(`${BASE_URL}`,{params:{noLogin: true}});
}; 
export const queryList_svc = (data:any): Promise<IPageResponse> => {
    return request.post(`${BASE_URL}`,data);
}; 

export const del_svc = (id:number): Promise<IResponse> => {
    return request.delete(`${BASE_URL}/${id}`, {});
}; 