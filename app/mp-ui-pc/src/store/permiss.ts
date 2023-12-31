import { defineStore } from 'pinia';
import { Storage } from '../store/Storage';

const storage=new Storage()
interface ObjectList {
	[key: string]: string[];
}

export const usePermissStore = defineStore('permiss', {
	state: () => {
		const keys = storage.get('ms_keys');
		//console.info("permiss:",keys)
		return {
			key: keys ? JSON.parse(keys) : <string[]>[],
			defaultList: <ObjectList>{
				admin: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17'],
				user: ['1', '2', '3', '11', '13', '14', '15']
			}
		};
	},
	actions: {
		handleSet(val: string[]) {
			this.key = val;
		}
	}
});
