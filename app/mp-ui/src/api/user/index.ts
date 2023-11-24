import request  from '../../utils/request';
const BASE_URL="/api/user"
export const queryList_svc = (data:IpageParam): Promise<IPageResponse> => {
    return request.post(`${BASE_URL}/list`,data);
};

export const exist_svc = ( userName:string): Promise<IResponse> => {
    return request.post(`${BASE_URL}/exist/${userName}`, {});
};
export const add_svc = ( data:any): Promise<IResponse> => {
    return request.post(`${BASE_URL}/add`, data);
};
export const edit_svc = (id:number,data:any): Promise<IResponse> => {
    return request.post(`${BASE_URL}/edit/${id}`, data);
};
export const del_svc = (id:number): Promise<IResponse> => {
    return request.post(`${BASE_URL}/del/${id}`, {});
};
export const retsetPwd_svc = (data:any): Promise<IResponse> => {
    return request.post(`${BASE_URL}/reset`, data);
};
export const changePwd_svc = (data:any): Promise<IResponse> => {
    return request.post(`${BASE_URL}/changePwd`, data);
};
export const currentUser_svc = ( ): Promise<IResponse> => {
    return request.get(`${BASE_URL}/currentUser`);
};

export const get_user_name_List_svc =( ): Promise<IResponse> => {
    return request.get(`${BASE_URL}/userList`);
};