import request  from '../../utils/request';
const BASE_URL="/api/biz/device" 



export const list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  