import { createServiceInstance, type IResponse, IEnumSelect } from 'co6co';
import { create_svc, create_association_svc } from '../base';
const base_URL = '/api/users';

export default create_svc(base_URL);

const association_service = create_association_svc(base_URL);
export { association_service };

export const exist_post_svc = (data: {
	userName: string;
	id: number;
}): Promise<IResponse<boolean>> => {
	return createServiceInstance().post(`${base_URL}/exist`, data);
};
export const retsetPwd_svc = (data: any): Promise<IResponse> => {
	return createServiceInstance(5000, false).post(`${base_URL}/reset`, data);
};
export const currentUser_svc = (): Promise<IResponse> => {
	return createServiceInstance().get(`${base_URL}/currentUser`);
};
export const changePwd_svc = (data: any): Promise<IResponse> => {
	return createServiceInstance().post(`${base_URL}/changePwd`, data);
};
export const ticket_svc = (code: string): Promise<IResponse> => {
	return createServiceInstance(5000, false).get(`${base_URL}/ticket/${code}`);
};

export interface UserType {
	username: string;
	password: string;
	role: string;
	roleId: string;
	permissions: string | string[];
}

export interface UserLogin {
	userName: string;
	password: string;
	verifyCode: string;
}
export const login_svc = (
	data: UserLogin,
	timeout: number = 5000,
	tip: boolean = false
): Promise<IResponse> => {
	return createServiceInstance(timeout, tip).post(`${base_URL}/login`, data);
};

export const get_state_svc = (): Promise<IResponse<IEnumSelect[]>> => {
	return createServiceInstance().post(`${base_URL}/status`);
};
export const get_category_svc = (): Promise<IResponse<IEnumSelect[]>> => {
	return createServiceInstance().post(`${base_URL}/category`);
};
