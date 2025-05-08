import {
	createServiceInstance,
	type IResponse,
	IEnumSelect,
	HttpContentType,
} from 'co6co';
import {
	createStreamServiceInstance,
	create_svc,
	create_association_svc,
} from '@/api/base';
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
export const retsetPwd_svc = (data: {
	userName: string;
	password: string;
}): Promise<IResponse> => {
	return createServiceInstance(5000, false).post(`${base_URL}/reset`, data);
};
export const currentUser_svc = (): Promise<
	IResponse<{ remark: string; avatar: string; userName: string }>
> => {
	return createServiceInstance().get(`${base_URL}/currentUser`);
};
export const changePwd_svc = (data: {
	newPassword: string;
	oldPassword: string;
	remark: string;
}): Promise<IResponse> => {
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

export const get_user_avatar = (): Promise<{ data: Blob }> => {
	return createStreamServiceInstance().get(`${base_URL}/avatar`, {
		responseType: 'blob',
	});
};
export const get_user_avatar2 = async (): Promise<Blob> => {
	try {
		const response = await createStreamServiceInstance().get(
			`${base_URL}/avatar`,
			{
				responseType: 'blob',
			}
		);
		return response.data;
	} catch (error) {
		console.error('Error fetching image:', error);
		throw error;
	}
};
export const put_user_avatar = (
	fromData: FormData
): Promise<IResponse<string>> => {
	return createServiceInstance(1500, true, HttpContentType.multipart).put(
		`${base_URL}/avatar`,
		fromData
	);
};
