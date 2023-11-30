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
interface memu {
	menuStates:Array<memu_state>
} 

export const wx_config_store = defineStore('wx_config',{
	state: () => {
		return {
			list: <ListItem[]>[],
			memu:<memu>{},
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
			if(res2.code==0) this.memu .menuStates= await res2.data 
			
		} ,
		getItem(v:string){ 
			if (v==null)return {name:"未设置",openId:""}
			return this.list.find(m=>m.openId=== v ) 
		},
		getMenuStateItem(v:number){  
			if (v==null)return {key:"未设置",label:"未设置",value:-1}
			return this.memu.menuStates.find(m=>m.value=== v )  
			
		}
	}
}) 