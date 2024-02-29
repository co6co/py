import { defineStore } from 'pinia';
import * as d from './types/devices';
import * as s from '../api/server';
import {type AlarmItem} from "../components/biz"

interface Object {
	[key: string]: any;
}
interface XssConfig {
	ip: string;
	port: number;
}
export const useAppDataStore = defineStore('app_Data', {
	state: () => {
		/*
		const keys = localStorage.getItem('app_Data');
		return {
			key: keys ? JSON.parse(keys) : <string[]>[],
		};*/
		return {
			data: {
				row: <AlarmItem>{},
				xssConfig: <XssConfig>{},
			},
		};
	},
	actions: {
		setState(val: any) {
			this.data.row = val;
		},
		getState() {
			return this.data.row;
		},
		async setXssConfig() { 
			const res = await s.get_xss_config_svc();
			if (res.code == 0) this.data.xssConfig = res.data;
			else console.warn('getXssConfig',res.message) 
		},
	},
});
