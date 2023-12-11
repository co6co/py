import request  from '../../utils/request';
const BASE_URL="/api/biz/device/camera" 


/* 树上获取所有的相机设备*/
export const list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  