import request  from '../../utils/request';
const BASE_URL="/api/topic"
export const query_svc = (category:string, code:String|number): Promise<IPageResponse<{topic:string}>> => {
    return request.get(`${BASE_URL}/${category}/${code}`);
};