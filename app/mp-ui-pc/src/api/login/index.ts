import request  from '../../utils/request'; 
import type { UserLogin } from './modle' 

export const login_svc = (data: UserLogin): Promise<IResponse> => {
    return request.post('/api/user/login',data,{responseType: 'json'}) 
}
  