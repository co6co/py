const base_URL = '/api/menu';
import { create_tree_svc } from '../base';
import { createServiceInstance, type IEnumSelect, type IResponse } from 'co6co';

const services = create_tree_svc(base_URL);
export default services;
export const get_category_svc = (): Promise<IResponse<IEnumSelect[]>> => {
	return createServiceInstance().post(`${base_URL}/category`);
};
export const get_state_svc = (): Promise<IResponse<IEnumSelect[]>> => {
	return createServiceInstance().post(`${base_URL}/status`);
};
import { MenuCateCategory } from '@/hooks/useMenuSelect';
export interface IMenuOne {
	id: number;
	name: string;
	component: string;
	category: MenuCateCategory;
	code: string;
	status: number;
}
export const get_one_svc = (id: number): Promise<IResponse<IMenuOne>> => {
	return createServiceInstance().get(`${base_URL}/${id}`);
};

export const batch_add_svc = (
	data: Array<any>
): Promise<IResponse<IResponse>> => {
	return createServiceInstance().post(`${base_URL}/batch`, data);
};
