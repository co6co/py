import { routeHook } from '@/hooks';
/**
 * 不对外导出
 */
export enum ConfigCodes {
	//配置CODE
	FILE_MGR_CODE = 'SYS_FILE_MANAGE',
}

export const defaultViewFeatures = {
	add: routeHook.ViewFeature.add,
	edit: routeHook.ViewFeature.edit,
	del: routeHook.ViewFeature.del,
};
