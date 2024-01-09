import request from '../../utils/request';
const BASE_URL = '/api/biz/site';

export const select_svc = ( ): Promise<IPageResponse> => {
	return request.get(`${BASE_URL}` );
};


//站点列表包括 相机 box
export const list_svc = (data: any): Promise<IPageResponse> => {
	return request.post(`${BASE_URL}`, data);
};

//站点列表 
export const list2_svc = (data: any): Promise<IPageResponse> => {
	return request.patch(`${BASE_URL}`, data);
};

export const getDetailInfo=(id:number ,category:string): Promise<IPageResponse> => { 
    return request.get(`${BASE_URL}/${id}?category=${category}` );
}; 
export const add_site_svc = (data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_URL}`,data);
}; 
export const edit_site_svc = (id:number,data:any): Promise<IPageResponse> => { 
    return request.put(`${BASE_URL}/${id}`,data);
}; 