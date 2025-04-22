import { createServiceInstance, type IResponse, type ISelect } from 'co6co';
import { create_svc } from 'co6co-right';

const base_URL = '/api/tasks/task';
const base_s_URL = '/api/tasks/task/sched';
const base_cron_URL = '/api/tasks/code/cron/test';

const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } =
	create_svc(base_URL);
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc };

export const exe_once_svc = (id: number): Promise<IResponse<string>> => {
	return createServiceInstance().put(`${base_s_URL}/${id}`);
};
export const get_select_svc = (
	category: number
): Promise<IResponse<ISelect[]>> => {
	return createServiceInstance().get(`${base_URL}/select/${category}`);
};
export const exe_sched_svc = (id: number): Promise<IResponse> => {
	return createServiceInstance().post(`${base_s_URL}/${id}`);
};
export const stop_sched_svc = (id: number): Promise<IResponse> => {
	return createServiceInstance().delete(`${base_s_URL}/${id}`);
};
export const get_next_exec_time_svc = (id: number): Promise<IResponse> => {
	return createServiceInstance().patch(`${base_s_URL}/${id}`);
};
export const test_cron_svc = (cron: string): Promise<IResponse<boolean>> => {
	return createServiceInstance().get(`${base_cron_URL}?cron=${cron}`);
};
export const test_cron2_svc = (cron: string): Promise<IResponse<boolean>> => {
	return createServiceInstance().post(`${base_cron_URL}`, { cron: cron });
};
