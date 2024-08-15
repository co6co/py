import { h, resolveComponent, VNodeChild } from 'vue';
/**
 *  解析组件名
 * @param componentName 组件名
 * @returns Vnode
 */
export const paraseNode = (componentName: string): VNodeChild => {
	try {
		const vnode = h(resolveComponent(componentName));
		return vnode;
	} catch (e) {
		console.warn(`未能解析组件：${componentName}`);
	}
};
