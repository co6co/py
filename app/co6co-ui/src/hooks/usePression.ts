import { Pinia } from 'pinia';
import { ConstObject } from '../constants';
import createPermissDirective from '../directives/permiss';
//import type { App } from '@vue/runtime-core';
import { App } from 'vue';
import { piniaInstance } from '../';

/**
 * 一个插件可以是一个拥有 install() 方法的对象，
 * 也可以直接是一个安装函数本身。安装函数会接收到安装它的应用实例和传递给 app.use() 的额外选项作为参数：
 * install(app, options) {
    // 配置此应用
  }
 * @param app
 */
// 定义插件接口
export interface IPluginOptions {
	instance?: Pinia;
}
// 创建插件
export const co6coPlugin = {
	install: (app: App, options: IPluginOptions) => {
		// 在这里使用 Pinia 实例
		//const pinia = options.instance;

		// 添加全局方法或其他逻辑
		//app.config.globalProperties.$myMethod = () => {
		//	console.log('Called myMethod');
		//};
		if (options.instance) app.use(options.instance);
		else app.use(piniaInstance);
		//app.config.globalProperties.$pinia 需要检测下.$pinia 是否存在
		const { permissDirective, nonPermissDirective } =
			createPermissDirective(piniaInstance);
		app.directive(ConstObject.getPermissValue(), permissDirective);
		app.directive(ConstObject.getNonPermissValue(), nonPermissDirective);
	},
};
/**
 * 舍弃 ->
 * use 要求必须返回个对象
 * @param app
 * @param options
 * @returns
 */
//const usePermiss = (app: App, options: { instance: Pinia }): any => {
//	try {
//		app.use(piniaInstance);
//		const { permissDirective, nonPermissDirective } =
//			createPermissDirective(piniaInstance);
//		app.directive(ConstObject.getPermissValue(), permissDirective);
//		app.directive(ConstObject.getNonPermissValue(), nonPermissDirective);
//
//		//挂在个全局方法
//		//Vue.prototype.postRequest = postRequest; /for vue2
//		//app.config.globalProperties.useDict = useDict //vue3
//		//组件中使用
//		/**
//		 * import { getCurrentInstance } from 'vue'
//		 * const { proxy, appContext } = getCurrentInstance()
//		 * appContext.config.globalProperties.$xxxx
//		 * appContext.config.globalProperties.xxxx()
//		 * proxy.$xxx
//		 * proxy.xxx()
//		 */
//	} catch (e) {
//		console.error(`增加指令:${ConstObject.getPermissValue()}失败!,Error:${e}`);
//	}
//	return {};
//};
