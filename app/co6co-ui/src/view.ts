import { moduleName, version } from '../package.json';
// 定义多个函数签名
function getViewPath(viewName: string): string;
function getViewPath(viewName: string, moduleName: string): string;
/**
 * 获取视图路径 
 * 参数格式化
 * @param viewName 视图名称
 * @param moduleName 模块名称
 * @returns 视图路径
 */
function getViewPath(...args: string[]): string {
	if (args.length == 1) {
		return `/views/${moduleName}/${args[0]}.vue`;
	} else if (args.length >= 2) {
		if (args[1]=="..") {
			//主模块不返回后缀，因为后缀有的是.vue，有的是.tsx 
			//不好分辨
			return `../views/${args[0]}`;
		}else{
			return `/views/${args[1]}/${args[0]}.vue`;
		}
	} else {
		throw `${args}不能为空!`;
	}
}
export { moduleName, getViewPath, version };
