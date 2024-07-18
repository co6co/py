import 'co6co/css/tables.css';
export * from './userTable';
export * from './menuTreeView';
export * from './roleView';
export * from './userGroupTreeView';
export * from './configView';
export * from './dictTypeView';
export * from './dictView';

import { moduleName } from '../../package.json';
export const getViewPath = (viewName: string) => {
	return `/views/${moduleName}/${viewName}.vue`;
};
