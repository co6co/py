import api from '@/services/api';

// TS 类型定义
export interface LoginParams {
  phone: string;
  code: string;
}

export interface UserInfo {
  id: number;
  name: string;
  phone: string;
  token: string;
}

 
// 登录
export const loginApi = (data: { phone: string; code: string }) => {
  return api.post<{ data: UserInfo }>('/user/login', data);
};

// 获取用户信息
export const getUserInfoApi = () => {
  return api.get<{ data: UserInfo }>('/user/info');
};