import { ViewFeature } from '@/constants';
/**
 * 不对外导出
 */
export enum ConfigCodes {
	//配置CODE
	FILE_MGR_CODE = 'SYS_FILE_MANAGE',
}

export const defaultViewFeatures = {
	add: ViewFeature.add,
	edit: ViewFeature.edit,
	del: ViewFeature.del,
};
