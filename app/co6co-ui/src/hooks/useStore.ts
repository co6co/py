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
			this.Config[routerKey] = router;
		},
		clearConfig() {
			this.Config = {};
		},
		setViews(views: ViewObjects) {
			Object.keys(views).forEach((key) => {
				this.ViewObject[key] = views[key];
			});
		},
		appendView(key: string, view: any) {
			// Vue received a Component that was made a reactive object. This can lead to unnecessary performance overhead and should be avoided by marking the component with `markRaw` or using `shallowRef` instead of `ref
			this.ViewObject[key] = markRaw(view);
		},
		appendViews(model: string, views: ViewObjects) {
			Object.keys(views).forEach((key) => {
				const path = getViewPath(key, model);
				this.appendView(path, markRaw(views[key]));
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
