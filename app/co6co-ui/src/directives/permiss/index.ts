import { usePermissStore } from './hook';
import type { Pinia } from 'pinia';
import type { DirectiveBinding, ObjectDirective } from 'vue';

const createPermissDirective = (pinia?: Pinia) => {
	const store = usePermissStore(pinia);
	const permissDirective: ObjectDirective = {
		mounted(el: HTMLElement, binding: DirectiveBinding) {
			if (!store.includes(String(binding.value))) {
				el['hidden'] = true;
			}
		},
	};
	/**
	 * 对 允许指令取反
	 */
	const nonPermissDirective: ObjectDirective = {
		mounted(el: HTMLElement, binding: DirectiveBinding) {
			if (store.includes(String(binding.value))) {
				el['hidden'] = true;
			}
		},
	};
	return { permissDirective, nonPermissDirective };
};

export default createPermissDirective;
export { usePermissStore };
