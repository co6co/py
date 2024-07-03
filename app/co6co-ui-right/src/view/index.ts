import 'co6co/css/tables.css';
export * from './userTable';
export * from './menuTreeView';
export * from './roleView';
export * from './userGroupTreeView';
import { moduleName } from '../../package.json';
export const getViewPath = (viewName: string) => {
	return `/views/${moduleName}/${viewName}.vue`;
};
