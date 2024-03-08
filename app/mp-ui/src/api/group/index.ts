import request from '../../utils/request';
const BASE_URL = '/api/user/group';
export const select_svc = ( ): Promise<IPageResponse> => {
	return request.get(`${BASE_URL}` );
};
