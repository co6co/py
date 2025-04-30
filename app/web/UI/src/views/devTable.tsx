import { defineComponent, VNodeChild } from 'vue'

import DevTable from '@/components/dev/devTable'

import * as api from '@/api/dev'
export default defineComponent({
  setup(prop, ctx) {
    //:page reader
    const rander = (): VNodeChild => {
      return <DevTable dataApi={api.get_table_svc} allowImport hasOpertion></DevTable>
    }
    return rander
  } //end setup
})
