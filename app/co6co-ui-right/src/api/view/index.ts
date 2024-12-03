import { create_tree_svc } from '../base';
const base_URL = '/api/view';

const { get_select_tree_svc } = create_tree_svc(base_URL);
export const view_route_svc = (key?: number | string) => {
	return get_select_tree_svc(key, 5000, false);
};
