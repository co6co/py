const base_URL = '/api/dict';

import { create_svc } from '../base';
import { DictSelectType } from './dictType';
import { createServiceInstance, type IEnumSelect, type IResponse } from 'co6co';
import { DictShowCategory } from '@/constants';

const { add_svc, edit_svc, del_svc, get_table_svc } = create_svc(base_URL);
export { add_svc, edit_svc, del_svc, get_table_svc };

export const get_dict_state_svc = (): Promise<IResponse<IEnumSelect[]>> => {
	return createServiceInstance().get(`${base_URL}`);
};

export interface IQueryDictSelectParam {
	dictTypeCode?: string; // 字典类型编码
	dictTypeId?: number; // 字典类型id  两属性不能同时为空
	category: DictShowCategory;
	parentId?: number;
}
/**
 * 获取字典的选择
 * @param dictTypeCode
 * @returns
 */
export const get_dict_select_svc = ( data :IQueryDictSelectParam): Promise<IResponse<DictSelectType[]>> => { 
	return createServiceInstance().post(`${base_URL}/select`,data);
};
