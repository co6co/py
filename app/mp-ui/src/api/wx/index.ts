import request  from '../../utils/request';
const BASE_URL="/api/wx"
const BASE_MENU_URL=`${BASE_URL}/menu`
const BASE_CONFIG_URL=`${BASE_URL}/config`

const BAE_PAGE_oauth2=`${BASE_URL}/oauth2` //页面认证
//菜单URL
export const get_menu_svc = (): Promise<IPageResponse> => {   
    return request.get(`${BASE_MENU_URL}`);
};  
export const list_menu_svc = (data:any): Promise<IPageResponse> => {   
    return request.post(`${BASE_MENU_URL}`,data);
};  
export const add_menu_svc = (data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_MENU_URL}`,data);
}; 
export const edit_menu_svc = (id:number,data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_MENU_URL}/${id}`,data);
}; 
export const del_menu_svc = (id:number): Promise<IPageResponse> => { 
    return request.delete(`${BASE_MENU_URL}/${id}`);
}; 
export const push_menu_svc = (id:number): Promise<IPageResponse> => { 
    return request.patch(`${BASE_MENU_URL}/${id}`);
}; 
export const pull_menu_svc = (id:number): Promise<IPageResponse> => { 
    return request.get(`${BASE_MENU_URL}/${id}`);
}; 

export const get_config_svc = (): Promise<IPageResponse> => { 
    return request.get(`${BASE_CONFIG_URL}`);
};  

export const get_oauth_svc = (): Promise<IPageResponse> => { 
    let param={code: "code",
        appid: "appid",
        url: window.location.href}
    return request.get(`${BAE_PAGE_oauth2}`);
}; 
