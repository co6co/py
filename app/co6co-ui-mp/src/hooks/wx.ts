import { piniaInstance } from 'co6co';
import { get_config_svc, get_menu_svc, type IMenuState } from '@/api/mp/menu';
import { defineStore } from 'pinia';

interface ListItem {
	name: string;
	openId: string;
}
const wx_config_store = defineStore('wx_config', {
	state: () => {
		return {
			list: [] as ListItem[],
			memuConfig: {} as IMenuState,
		};
	},
	getters: {
		show: (state) => {
			return state.list.length > 0;
		},
		nameList: (state) => {
			return state.list.map((item) => item.name);
		},
	},
	actions: {
		async refesh() {
			await this.getConfig(true);
			await this.getMemuConfig(true);
		},
		async getConfig(refesh: boolean = false) {
			if (this.list.length == 0 || refesh) {
				const res = await get_config_svc();
				this.list = res.data;
			}
		},
		async getMemuConfig(refesh: boolean = false) {
			if (!this.memuConfig.menuStates || refesh) {
				const res2 = await get_menu_svc();
				this.memuConfig = await res2.data;
			}
		},
		getItem(v: string) {
			if (v == null) return { name: '未设置', openId: '' };
			return this.list.find((m) => m.openId === v);
		},
		getMenuStateItem(v: number) {
			if (v == null) return { key: '未设置', label: '未设置', value: -1 };
			return this.memuConfig.menuStates.find((m) => m.value === v);
		},
	},
});

export const get_store = () => {
	const store = wx_config_store(piniaInstance);
	return store;
};
