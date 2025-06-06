import { defineStore } from 'pinia';
//import { getCurrentInstance } from 'vue';
//const instance = getCurrentInstance();

import type { Plugin } from 'vue';

// 组件注册表项类型
interface ComponentRegistryItem {
	identifier: string; // 用户自定义的唯一标识
	plugin: Plugin; // Vue 插件对象
	options?: Record<string, any>; // 可选配置项
}

export const useComponentStore = defineStore('component', {
	state: () => ({
		registry: new Map<string, ComponentRegistryItem>(), // 使用 Map 保证键唯一性
	}),

	actions: {
		/**
		 * 注册组件
		 * @param identifier 用户自定义的组件标识
		 * @param plugin 组件插件对象
		 * @param options 可选配置项
		 */
		registerComponent(
			identifier: string,
			plugin: Plugin,
			options?: Record<string, any>
		) {
			if (this.registry.has(identifier)) {
				console.warn(`组件标识 "${identifier}" 已存在，将覆盖原有注册`);
			}
			this.registry.set(identifier, { identifier, plugin, options });
		},

		/**
		 * 通过标识获取组件
		 * @param identifier 组件标识
		 */
		getComponent(identifier: string) {
			return this.registry.get(identifier);
		},

		/**
		 * 卸载组件
		 * @param identifier 组件标识
		 */
		unregisterComponent(identifier: string) {
			this.registry.delete(identifier);
		},

		/**
		 * 应用所有已注册的组件
		 * @param app Vue 应用实例
		 * 似乎用不到
		 */
		//applyAllComponents(app: App) {
		//	this.registry.forEach((item) => {
		//		app.use(item.plugin, item.options);
		//	});
		//},
	},

	getters: {
		/** 获取所有已注册的组件标识 */
		allIdentifiers: (state) => Array.from(state.registry.keys()),
	},
});
