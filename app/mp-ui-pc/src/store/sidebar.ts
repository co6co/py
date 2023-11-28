import { defineStore } from 'pinia';

export const useSidebarStore = defineStore('sidebar', {
	state: () => {
		return {
			//是否收起
			collapse:false
		};
	},
	getters: {},
	actions: {
		handleCollapse() { 
			this.collapse = !this.collapse;
		}
	}
});
