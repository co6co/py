import { defineComponent, onMounted, VNodeChild } from 'vue'
import { ViewFeature, routeHook } from 'co6co-right'

import DevTable from '@/components/dev/devTable'
import * as api from '@/api/dev'
export const features = {
  add: ViewFeature.add,
  edit: ViewFeature.edit,
  del: ViewFeature.del
}
export default defineComponent({
  setup(prop, ctx) {
    const { getPermissKey } = routeHook.usePermission()
    onMounted(() => {
      console.info('getPermissKey(features.edit)=>', getPermissKey(features.edit))
      console.info('getPermissKey(features.add)=>', getPermissKey(features.add))
    })
    //:page reader
    const rander = (): VNodeChild => {
      return <DevTable dataApi={api.get_table_svc} allowImport hasOpertion></DevTable>
    }
    return rander
  } //end setup
})
