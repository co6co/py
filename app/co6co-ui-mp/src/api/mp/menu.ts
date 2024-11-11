import {
	createServiceInstance,
	type IPageResponse,
	type IEnumSelect,
} from 'co6co';

const BASE_URL = '/api/wx';
const BASE_MENU_URL = `${BASE_URL}/menu`;
const BASE_CONFIG_URL = `${BASE_URL}/config`;
const BAE_PAGE_oauth2 = `${BASE_URL}/oauth2`; //页面认证
 
export interface IMenuState {
	menuStates: IEnumSelect[];
}

export interface IMenuConfig {
	openId: string;
	name: string;
}

//菜单URL
export const get_menu_svc = (): Promise<IPageResponse<IMenuState>> => {
	return createServiceInstance().get(`${BASE_MENU_URL}`);
};
export const list_menu_svc = (data: any): Promise<IPageResponse> => {
	return createServiceInstance().post(`${BASE_MENU_URL}`, data);
};
export const add_menu_svc = (data: any): Promise<IPageResponse> => {
	return createServiceInstance().put(`${BASE_MENU_URL}`, data);
};
export const edit_menu_svc = (
	id: number,
	data: any
): Promise<IPageResponse> => {
	return createServiceInstance().put(`${BASE_MENU_URL}/${id}`, data);
};
export const del_menu_svc = (id: number): Promise<IPageResponse> => {
	return createServiceInstance().delete(`${BASE_MENU_URL}/${id}`);
};
export const push_menu_svc = (id: number): Promise<IPageResponse> => {
	return createServiceInstance().patch(`${BASE_MENU_URL}/${id}`);
};
export const pull_menu_svc = (id: number): Promise<IPageResponse> => {
	return createServiceInstance().get(`${BASE_MENU_URL}/${id}`);
};
/**
 * 重置公众号ID
 * @param id 当前菜单ID
 * @returns
 */
export const reset_menu_svc = (id: number): Promise<IPageResponse> => {
	console.warn('未实现');
	return createServiceInstance().get(`${BASE_MENU_URL}/${id}`);
};
export const get_config_svc = (): Promise<IPageResponse<IMenuConfig[]>> => {
	return createServiceInstance().get(`${BASE_CONFIG_URL}`);
};
export const get_oauth_svc = (): Promise<IPageResponse> => {
	//let param = { code: 'code', appid: 'appid', url: window.location.href }
	return createServiceInstance().get(`${BAE_PAGE_oauth2}`);
};
