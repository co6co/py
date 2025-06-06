//import { type ConfigProviderContext, provideGlobalConfig } from 'element-plus'
import { INSTALLED_KEY, PiniaInstanceKey } from './constants';
import { version } from '../package.json';
//import type { App, Plugin } from '@vue/runtime-core';
import type { App, Plugin } from 'vue';
import { createPinia } from 'pinia';
export const Installer = (
	components: Plugin[] = [],
	pluginName: string | symbol,
	installbck?: (app: App, options?: any) => void
) => {
	const install = (app: App, options?: any) => {
		if (app[pluginName]) return;
		app[pluginName] = true;
		if (components && components.length > 0)
			components.forEach((c) => app.use(c));
		//if (options) provideGlobalConfig(options, app, true)
		if (installbck) installbck(app, options);
	};
	return {
		install,
	};
};

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
		Installer(components, INSTALLED_KEY).install(app);
		app[PiniaInstanceKey] = piniaInstance;
	};
	return {
		version,
		install,
		piniaInstance,
	};
};
