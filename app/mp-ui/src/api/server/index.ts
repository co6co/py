import request  from '../../utils/request';
const BASE_URL="/api/server/getxssConfig" 
 
//获取xss 的服务器配置
export const get_xss_config_svc =(): Promise<IResponse> => {
    return request.post(`${BASE_URL}`);
};