import request  from '../../utils/request';
const device_BASE_URL="/api/biz/device" 
const BASE_URL="/api/biz/device/camera" 


export const device_type_svc = ( ): Promise<IPageResponse> => {   
    return request.get(`${device_BASE_URL}` );
};  
/* 树上获取所有的相机设备*/
export const list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  
export const get_poster_svc = (id:number): Promise<IPageResponse> => {   
    return request.get(`${BASE_URL}/${id}`);
};   