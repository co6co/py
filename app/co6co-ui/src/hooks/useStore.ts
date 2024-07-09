import { defineStore } from 'pinia';
import { piniaInstance } from '../index';
type ConfigValue = string | number | boolean | any;
interface Config {
	[key: string]: ConfigValue;
}
interface ViewObjects {
	[key: string]: any;
}
const baseUrl = 'baseURL';

export const useStore = defineStore('co6co_store', {
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
		views: (state) => {
			return state.ViewObject;
		},
	},
	actions: {
		config(key: string, value: ConfigValue) {
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
		setViews(views: any) {
			Object.keys(views).forEach((key) => {
				this.ViewObject[key] = views[key];
			});
		},
		setView(key: string, view: any) {
			this.ViewObject[key] = view;
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
