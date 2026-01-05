const base_URL = '/api/config';
import { create_svc } from '../base';
import { createServiceInstance, type IResponse } from 'co6co';

const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } =
	create_svc(base_URL);
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc };

export interface IConfig {
	name: string;
	code: string;
	value: string;
	remark: string;
}
/**
 * 获取字典的选择
 * @param dictTypeCode
 * @returns
 */
export const get_config_svc = (code: string): Promise<IResponse<IConfig>> => {
	return createServiceInstance().get(`${base_URL}/${code}`);
};
/**
 * 获取UI配置
 * @returns 配置不需要认证
 */
export const get_ui_config_svc = ( ): Promise<IResponse<Record<string, any>>> => {
	return createServiceInstance(5000,false).post(`${base_URL}/ui`);
	
};
