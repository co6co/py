import request  from '../../utils/request';
const BASE_URL="/api/biz/device/camera" 
const device_BASE_URL="/api/biz/device"  


export const dev_type_svc = ( ): Promise<IPageResponse> => {   
    return request.get(`${device_BASE_URL}` );
};  
//应该删除
export const dev_list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${device_BASE_URL}`,data);
};   
/* 树上获取所有的相机设备*/
export const list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  
export const get_poster_svc = (id:number): Promise<IPageResponse> => {   
    return request.get(`${BASE_URL}/${id}`);
};   
export const add_camera_svc = (data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_URL}`,data);
}; 
export const edit_camera_svc = (id:number,data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_URL}/${id}`,data);
}; 
export const del_camera_svc = (id:number ): Promise<IPageResponse> => { 
    return request.delete(`${BASE_URL}/${id}`);
}; 