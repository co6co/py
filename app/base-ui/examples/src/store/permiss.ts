import { defineStore } from 'pinia'
export const usePermissStore = defineStore('permiss', {
  state: () => {
    //const keys = getAllPermissionKeys();
    return {
      key: <string[]>[]
      /*
			defaultList: <ObjectList>{
				admin: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','admin','user'],
				user: ['1', '2', '3', '11', '13', '14', '15','user']
			}
			*/
    }
  },
  actions: {
    set(val: string[]) {
      this.key = val
    },
    includes(val: string) {
      return this.key.includes(val)
    }
  }
})
