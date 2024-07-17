import { Pinia } from 'pinia';
import { ConstObject } from '../constants';
import createPermissDirective from '../directives/permiss';
import type { App } from '@vue/runtime-core';
import { piniaInstance } from '../';

/**
 * 一个插件可以是一个拥有 install() 方法的对象，
 * 也可以直接是一个安装函数本身。安装函数会接收到安装它的应用实例和传递给 app.use() 的额外选项作为参数：
 * install(app, options) {
    // 配置此应用
  }
 * @param app
 */

export const installPermissDirective = (
	app: App,
	option?: { instance: Pinia }
) => {
	try {
		console.info('installPermissDirective:', option);
		app.use(piniaInstance);
		app.directive(
			ConstObject.getPermissValue(),
			createPermissDirective(piniaInstance)
		);
	} catch (e) {
		console.error(`增加指令:${ConstObject.getPermissValue()}失败!,Error:${e}`);
	}
};
