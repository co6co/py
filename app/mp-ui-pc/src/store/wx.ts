import { get_config_svc } from '../api/wx';
import { get_menu_svc } from '../api/wx';
import { defineStore } from 'pinia'; 

interface ListItem {
	name: string;
	openId: string; 
} 
interface memu_state{
	key:string,label:string,value:number
}
interface memuConfig {
	menuStates:memu_state[] //Array<memu_state>
}  
export const wx_config_store = defineStore('wx_config',{
	state: () => {
		return {
			list: <ListItem[]>[],
			memuConfig:<memuConfig>{},
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
			const res2 =await get_menu_svc() 
			if(res2.code==0) {
				const data=await res2.data
				this.memuConfig=data
			}
		} ,
		getItem(v:string){ 
			if (v==null)return {name:"未设置",openId:""}
			return this.list.find(m=>m.openId=== v ) 
		},
		getMenuStateItem(v:number){  
			if (v==null)return {key:"未设置",label:"未设置",value:-1} 
			return this.memuConfig.menuStates.find(m=>m.value=== v )  
			
		}
	}
}) 