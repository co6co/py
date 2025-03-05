import { create_tree_svc } from '../base';
import { createServiceInstance, type IResponse } from 'co6co';
const base_URL = '/api/view';

const { get_select_tree_svc } = create_tree_svc(base_URL);
export const view_route_svc = (key?: number | string) => {
	return get_select_tree_svc(key, 5000, false);
};

export const get_session_svc = (): Promise<
	IResponse<{ data: string; expiry: number }>
> => {
	return createServiceInstance().post(base_URL);
};
