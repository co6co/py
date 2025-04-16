import { create_svc } from 'co6co-right';
import { createServiceInstance, type IResponse } from 'co6co';

const base_URL = '/api/tasks/dynamicCode';
const base_code_URL = '/api/tasks/code/test';

const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } =
	create_svc(base_URL);
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc };

export const test_code_svc = (code: string): Promise<IResponse<boolean>> => {
	return createServiceInstance().post(`${base_code_URL}`, { code: code });
};
/**
 * 测试 python 代码是否符合规定
 * @param code
 * @returns
 */
export const test_exe_code_svc = (code: string): Promise<IResponse<string>> => {
	return createServiceInstance().put(`${base_code_URL}`, { code: code });
};
export const exe_once_svc = (id: number): Promise<IResponse<string>> => {
	return createServiceInstance().put(`${base_URL}/run/${id}`);
};
