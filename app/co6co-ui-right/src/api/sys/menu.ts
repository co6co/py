const base_URL = '/api/menu';
import { create_tree_svc } from '../';
import request from '@/utils/request';
import * as api_type from 'co6co';

const services = create_tree_svc(base_URL);
export default services;
export const get_category_svc = (): Promise<
	api_type.IResponse<api_type.IEnumSelect[]>
> => {
	return request.post(`${base_URL}/category`);
};
export const get_state_svc = (): Promise<
	api_type.IResponse<api_type.IEnumSelect[]>
> => {
	return request.post(`${base_URL}/status`);
};
