import request  from '../../utils/request';
const BASE_URL="/api/biz/device" 
const Base_Category_URL="/api/biz/category"



export const list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  

export const category_list_svc = ( ): Promise<IPageResponse> => {   
    return request.get(`${Base_Category_URL}` );
};  

export const set_ligth_svc = ( data:any): Promise<IPageResponse> => {   
    return request.patch(`${BASE_URL}` ,data);
};  