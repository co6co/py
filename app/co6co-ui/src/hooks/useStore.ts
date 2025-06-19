import { defineStore } from 'pinia';
import { piniaInstance } from '../index';
import { getViewPath } from '../view';
import { TIView } from '@/constants';
type ConfigValue = string | number | boolean | any;
interface Config {
	[key: string]: ConfigValue;
}
interface ViewObjects {
	[key: string]: TIView;
}
const baseUrl = 'baseURL';
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
		clearConfig() {
			this.Config = {};
		},
		setViews(views: ViewObjects) {
			Object.keys(views).forEach((key) => {
				this.ViewObject[key] = views[key];
			});
		},
		appendView(key: string, view: any) {
			this.ViewObject[key] = view;
		},
		appendViews(model: string, views: ViewObjects) {
			Object.keys(views).forEach((key) => {
				const path = getViewPath(model, key);
				this.appendView(path, views[key]);
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
