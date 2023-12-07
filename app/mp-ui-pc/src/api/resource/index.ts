import request  from '../../utils/request';
const BASE_URL="/api/resource" 

export const get_resource =(uuid:string): Promise<IResponse> => {
    return request.get(`${BASE_URL}/${uuid}`);
};