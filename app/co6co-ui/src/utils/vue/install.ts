import { NOOP } from '@vue/shared';

import type { App, Directive } from 'vue';
import type { SFCInstallWithContext, SFCWithInstall } from './typescript';
// withInstall 函数的典型实现
/**
export function withInstall<T>(component: T, alias?: string) {
  const comp = component as any;
  comp.install = (app: App) => {
    app.component(comp.name || comp.displayName, component);
    if (alias) {
      app.config.globalProperties[alias] = component;
    }
  };
  return component as T & { install: (app: App) => void };
}*/
export const withInstall = <T, E extends Record<string, any>>(
	main: T,
	extra?: E
) => {
	(main as SFCWithInstall<T>).install = (app: App): void => {
		for (const comp of [main, ...Object.values(extra ?? {})]) {
			app.component(comp.name, comp);
		}
	};

	if (extra) {
		for (const [key, comp] of Object.entries(extra)) {
			(main as any)[key] = comp;
		}
	}
	return main as SFCWithInstall<T> & E;
};

export const withInstallFunction = <T>(fn: T, name: string) => {
	(fn as SFCWithInstall<T>).install = (app: App) => {
		(fn as SFCInstallWithContext<T>)._context = app._context;
		app.config.globalProperties[name] = fn;
	};

	return fn as SFCInstallWithContext<T>;
};

export const withInstallDirective = <T extends Directive>(
	directive: T,
	name: string
) => {
	(directive as SFCWithInstall<T>).install = (app: App): void => {
		app.directive(name, directive);
	};

	return directive as SFCWithInstall<T>;
};

export const withNoopInstall = <T>(component: T) => {
	(component as SFCWithInstall<T>).install = NOOP;

	return component as SFCWithInstall<T>;
};
