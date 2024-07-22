const base_URL = '/api/dict';

import { create_svc } from '../base';
import { DictSelect } from './dictType';
import { createServiceInstance, type IEnumSelect, type IResponse } from 'co6co';

const { add_svc, edit_svc, del_svc, get_table_svc } = create_svc(base_URL);
export { add_svc, edit_svc, del_svc, get_table_svc };

export const get_dict_state_svc = (): Promise<IResponse<IEnumSelect[]>> => {
	return createServiceInstance().get(`${base_URL}`);
};

/**
 * 获取字典的选择
 * @param dictTypeCode
 * @returns
 */
export const get_dict_select_svc = (
	dictTypeCode: string
): Promise<IResponse<DictSelect[]>> => {
	return createServiceInstance().get(`${base_URL}/${dictTypeCode}`);
};
