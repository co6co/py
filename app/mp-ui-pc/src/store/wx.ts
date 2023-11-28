import { get_config_svc } from '../api/wx';
import { defineStore } from 'pinia'; 

interface ListItem {
	name: string;
	openId: string; 
}

export const wx_config_store = defineStore('wx_config',{
	state: () => {
		return {
			list: <ListItem[]>[]
		};
	},
	getters: {
		show: state => {
			return state.list.length > 0;
		},
		nameList: state => {
			return state.list.map(item => item.name);
		}
	},
	actions: {
		async refesh( ) {
			const res =await get_config_svc() 
			if(res.code==0) this.list = await res.data 
		} ,
		getItem(v:string){ 
			if (v==null)return {name:"未设置",openId:""}
			return this.list.find(m=>m.openId=== v ) 
		}
	}
}) 