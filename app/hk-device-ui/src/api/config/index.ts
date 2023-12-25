import request  from '../../utils/request';
const BASE_URL="/api/sys/config"  

export const list_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_URL}`,data);
};  

 
export const add_config_svc = (data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_URL}`,data);
}; 
export const edit_config_svc = (id:number,data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_URL}/${id}`,data);
}; 
export const del_config_svc = (id:number ): Promise<IPageResponse> => { 
    return request.delete(`${BASE_URL}/${id}`);
}; 

 