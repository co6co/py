import { defineComponent, VNodeChild,onMounted } from 'vue'
import { routeHook } from 'co6co-right'

import DevTable ,{features}from '@/components/dev/devTable'
import * as api from '@/api/dev' 
export {features}
export default defineComponent({
  setup(prop, ctx) {
    const { getPermissKey } = routeHook.usePermission()
    onMounted(() => {
      console.info('getPermissKey(features.edit)=>', getPermissKey(features.edit))
      console.info('getPermissKey(features.add)=>', getPermissKey(features.add))
    })
    //:page reader
    const rander = (): VNodeChild => {
      return <DevTable dataApi={api.get_table_svc} allowImport hasOpertion /> 
    }
    return rander
  } //end setup
})
