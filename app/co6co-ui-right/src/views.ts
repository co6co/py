import { moduleName } from '../package.json';
import {
	UserTableView,
	RoleView,
	MenuTreeView,
	UserGroupTreeView,
	ConfigView,
	DictTypeView,
	DictView,
	FileManageView,
	PreviewView,
} from './view';
/**
 * 应用通过该对象获取所有的页面视图
 */
export const views = {
	UserTableView,
	RoleView,
	MenuTreeView,
	UserGroupTreeView,
	ConfigView,
	DictTypeView,
	DictView,
	FileManageView,
	PreviewView,
};

// 定义多个函数签名
function getViewPath(viewName: string): string;
function getViewPath(viewName: string, moduleName: string): string;
function getViewPath(...args: string[]): string {
	if (args.length == 1) {
		return `/views/${moduleName}/${args[0]}.vue`;
	} else if (args.length >= 2) {
		return `/views/${args[1]}/${args[0]}.vue`;
	} else {
		throw `${args}不能为空!`;
	}
}
export { moduleName, getViewPath };
