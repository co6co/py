import request from '../../utils/request';
const BASE_URL = '/api/biz/site';

export const list_svc = (data: any): Promise<IPageResponse> => {
	return request.post(`${BASE_URL}`, data);
};
