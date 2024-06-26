import { get_status_svc } from '../api/group'
import { defineStore } from 'pinia'

export const group_state_store = defineStore('boat_group_state_store', {
  state: () => {
    return {
      loaded: false,
      groupItems: [] as Array<optionItem>,
      postionItems: [] as Array<optionItem>,
      allowSetNumberGroup: [] as Array<string>,
      allowSetPriorityGroup: [] as Array<string>
    }
  },
  getters: {
    group: (state) => {
      return state.groupItems
    },
    postion: (state) => {
      return state.groupItems
    },
    allowGroupName: (state) => {
      return state.allowSetNumberGroup
    }
  },
  actions: {
    async refesh() {
      await this.getConfig(true)
    },
    async getConfig(refesh: boolean = false) {
      if (!this.loaded || refesh) {
        const res = await get_status_svc()
        if (res.code == 0) {
          this.groupItems = []
          this.groupItems.push({ key: '', label: '--分组类型--' })
          this.groupItems.push(...res.data.group)
          this.postionItems = []
          this.postionItems.push({ key: '', label: '--安装位置--' })
          this.postionItems.push(...res.data.postion)
          this.allowSetNumberGroup = []
          this.allowSetNumberGroup.push(...res.data.allowSetNumberGroup)
          this.allowSetPriorityGroup = []
          this.allowSetPriorityGroup.push(...res.data.allowSetPriorityGroup)
          this.loaded = true
        }
      }
    },
    getGroupItem(v: string) {
      if (v === null) return { key: '', value: undefined, label: '' }
      return this.groupItems.find((m) => m.key === v)
    },
    getPostionItem(v: number) {
      if (v === null) return { key: '', value: undefined, label: '' }
      return this.postionItems.find((m) => m.value === v)
    },
    statue2TagType(v?: number) {
      if (v === null) return 'info'
      switch (v) {
        case 0:
          return 'danger'
        //case 1:return 'primary'
        case 2:
          return 'success'
        case 3:
          return 'warning'
        default:
          return '' //primary
      }
    },
    //允许编码的 类型
    allowSetting(groupType: string) {
      return this.allowSetNumberGroup.indexOf(groupType) > -1
    },
    //允许设置优先级的 类型
    allowSetPriority(groupType: string) {
      return this.allowSetPriorityGroup.indexOf(groupType) > -1
    }
  }
})
