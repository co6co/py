import request  from '../../utils/request';
const BASE_URL="/api/alarm" 

 
export const list_menu_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  