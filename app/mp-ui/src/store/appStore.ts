import { defineStore } from 'pinia';
import * as d from './types/devices'

interface Object  {
	[key: string]: any;
}

export const useAppDataStore = defineStore('app_Data', {
	state: ( ) => {
		/*
		const keys = localStorage.getItem('app_Data');
		return {
			key: keys ? JSON.parse(keys) : <string[]>[],
		};*/
		return {
			data:{
				row:{}
				 
			}
		 }
	},
	actions: {
		setState(val:any) {
			this.data.row = val; 
		},
		getState(  ) {
			return this.data.row  ;
		}
	}
});
