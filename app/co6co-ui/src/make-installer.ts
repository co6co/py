//import { type ConfigProviderContext, provideGlobalConfig } from 'element-plus'
import { INSTALLED_KEY, PiniaInstanceKey } from './constants';
import { version } from '../package.json';

import type { App, Plugin } from '@vue/runtime-core';
import { createPinia } from 'pinia';
//确保全局唯一
/**
 * 只要有引入本项目就会执行
 *
 * @param components
 * @returns
 */
export const makeInstaller = (components: Plugin[] = []) => {
	const piniaInstance = createPinia();
	//加载所有的组件，但当应用不执行就不加载
	const install = (app: App /*, options?: ConfigProviderContext*/) => {
		//todo 这里到底会不会执行，有无效果/
		console.info('makeInstaller.install...');
		if (app[INSTALLED_KEY]) return;
		app[INSTALLED_KEY] = true;
		app[PiniaInstanceKey] = piniaInstance;
		components.forEach((c) => app.use(c));
		//if (options) provideGlobalConfig(options, app, true)
	};
	return {
		version,
		install,
		piniaInstance,
	};
};
