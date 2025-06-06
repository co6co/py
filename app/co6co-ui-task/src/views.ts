export { moduleName, version } from '../package.json';
import { TaskTableView, DynamicTableView } from './view';
/**
 * 应用通过该对象获取所有的页面视图
 */
export const views = {
	TaskTableView,
	DynamicTableView,
};
