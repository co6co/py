import request  from '../../utils/request';
const BASE_URL="/api/wx/menu"

 
export const queryList_svc = (data:any): Promise<IPageResponse> => {
    return request.post(`${BASE_URL}/list`,data);
}; 

export const update_svc = (id:number,data:any): Promise<IResponse> => {
    return request.post(`${BASE_URL}/boatPosNumber/${id}`,data);
}; 
export const get_status_svc = (): Promise<IPageResponse> => {
    return request.get(`${BASE_URL}/getStatus`,{params:{noLogin: true}});
}; 
export const get_one_svc = (id:number): Promise<IPageResponse> => {
    return request.post(`${BASE_URL}/one/${id}`,{params:{noLogin: true}});
}; 
