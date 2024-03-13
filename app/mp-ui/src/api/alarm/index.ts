import request  from '../../utils/request';
const BASE_URL="/api/biz/alarm" 
const BASE_Category_URL="/api/biz/alarm/category" 
import { type AlarmItem } from '../../components/biz'

export const list_svc = (data:any): Promise<IPageResponse<AlarmItem[]>> => {   
    return request.post(`${BASE_URL}`,data);
};  

//通过[id  |  uuid] 查询单条记录
export const get_one = (uidOrid:string|number): Promise<IResponse<AlarmItem>> => {   
    return request.post(`${BASE_URL}/${uidOrid}`);
}; 
export const alert_category_svc = (): Promise<IPageResponse> => {   
    return request.get(`${BASE_Category_URL}`);
};  