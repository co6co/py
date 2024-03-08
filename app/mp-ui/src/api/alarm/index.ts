import request  from '../../utils/request';
const BASE_URL="/api/biz/alarm" 
 
const BASE_Category_URL="/api/biz/alarm/category" 
export const list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  
export const alert_category_svc = (): Promise<IPageResponse> => {   
    return request.get(`${BASE_Category_URL}`);
};  