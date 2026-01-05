import { defineStore } from 'pinia';
import { piniaInstance } from '../index';
import { getViewPath } from '../view';
import { ViewComponent } from '@/constants';
import { Router } from 'vue-router';
import { markRaw } from 'vue';
type ConfigValue = string | number | boolean | any;
interface Config {
	[key: string]: ConfigValue;
}
interface ViewObjects {
	[key: string]: ViewComponent;
}
const baseUrl = 'baseURL';
const routerKey = 'Router';
const useStore = defineStore('co6co_store', {
	state: () => {
		return {
			ViewObject: <ViewObjects>{},
			Config: <Config>{},
		};
	},
	getters: {
		config: (state) => {
			return state.Config;
		},
		apiBaseUrl: (state) => {
			return state.Config[baseUrl] as string;
		},
		views: (state) => {
			return state.ViewObject;
		},
		router: (state) => {
			return state.Config[routerKey] as Router;
		},
	},
	actions: {
		setConfig(key: string, value: ConfigValue) {
			this.Config[key] = value;
		},
		getConfig(key: string) {
			return this.Config[key];
		},
		getBaseUrl() {
			return this.Config[baseUrl] as string;
		},
		setBaseUrl(url: string) {
			this.Config[baseUrl] = url;
		},

		setRouter(router: Router) {
			// 路由对象不应该被响应式化
			this.Config[routerKey] = markRaw(router);
		},
		clearConfig() {
			this.Config = {};
		},
		setViews(views: ViewObjects) {
			// 清空现有视图并设置新视图
			this.ViewObject = {};
			Object.entries(views).forEach(([key, view]) => {
				this.ViewObject[key] = markRaw(view);
			});
		},
		
		appendView(key: string, view: ViewComponent) {
			// 确保组件不被响应式化
			this.ViewObject[key] = markRaw(view);
		},
		appendViews(model: string, views: ViewObjects) {
			// 确保model是有效的字符串
			if (typeof model !== 'string' || !model.trim()) {
				console.warn('Invalid model name provided to appendViews');
				return;
			}
			
			Object.entries(views).forEach(([key, view]) => {
				const path = getViewPath(key, model);
				this.appendView(path, view);
			});
		},
		clearView() {
			this.ViewObject = {};
		},
	},
});

export const getStoreInstance = () => {
	const store = useStore(piniaInstance);
	return store;
};
