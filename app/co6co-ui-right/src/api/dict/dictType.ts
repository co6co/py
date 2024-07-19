const base_URL = '/api/dict/type';

import { create_svc } from '../base';
import { createServiceInstance, type ISelect, type IResponse } from 'co6co';

const { get_select_svc, add_svc, edit_svc, del_svc, get_table_svc, exist_svc } =
	create_svc(base_URL);

export { get_select_svc, add_svc, edit_svc, del_svc, get_table_svc, exist_svc };

export type DictSelect = ISelect & { value: string; desc: string };
/**
 * 获取字典的选择
 * @param dictTypeId
 * @returns
 */
export const get_dict_select_svc = (
	dictTypeId: number
): Promise<IResponse<DictSelect[]>> => {
	return createServiceInstance().get(`${base_URL}/${dictTypeId}`);
};

/**
 * 获取字典的列表
 * @param dictTypeId
 * @returns
 */
export const get_dict_table_svc = (dictTypeId: number): Promise<IResponse> => {
	return createServiceInstance().post(`${base_URL}/${dictTypeId}`);
};
