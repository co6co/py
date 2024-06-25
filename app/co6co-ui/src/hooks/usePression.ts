import { createPinia } from 'pinia';
import { ConstObject } from '../constants';
import createPermissDirective from '../directives/permiss';
import type { App } from '@vue/runtime-core';

export const usePermiss = (app: App) => {
	const pinia = createPinia();
	app.use(pinia);
	app.directive(ConstObject.getPermissValue(), createPermissDirective(pinia));
};
