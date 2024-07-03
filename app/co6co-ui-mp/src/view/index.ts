import { moduleName } from '../../package.json';
export const getViewPath = (viewName: string) => {
	return `/views/${moduleName}/${viewName}.vue`;
};

export * from './menuView';
