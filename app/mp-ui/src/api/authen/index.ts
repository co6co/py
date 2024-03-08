import request  from '../../utils/request'; 
import type { UserLogin } from './modle' 
import {type Token} from '../../store/types/user'
const BASE_URL="/api/user"

//用户名登录
export const login_svc = (data: UserLogin): Promise<IResponse<Token>> => {
    return request.post(`${BASE_URL}/login`,data,{responseType: 'json'}) 
} 
//票据登录
export const ticket_svc =(ticket:string ): Promise<IResponse<Token>> => {
    return request.post(`${BASE_URL}/ticket/${ticket}`);
};