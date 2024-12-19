import { computed, defineComponent, nextTick, VNodeChild, watch } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import {
  ElButton,
  ElInput,
  ElTableColumn,
  ElRow,
  ElCol,
  ElLink,
  ElTooltip,
  ElSelect,
  ElOption
} from 'element-plus'
import { Search, Edit, ArrowLeftBold, Refresh, Delete, UploadFilled } from '@element-plus/icons-vue'

import {
  routeHook,
  tableScope,
  TableView,
  TableViewInstance,
  Download,
  download_header_svc,
  deleteHook
} from 'co6co-right'

import { clc_svc, param } from '@/api/tool'
export default defineComponent({
  setup(prop, ctx) {
    const DATA = reactive<{
      category: number
      list: Array<number>
      dans?: Array<number>
      listStr: string
      dansStr: string
      result: Array<String>
    }>({ listStr: '', dansStr: '', category: 0, list: [], result: [] })
    //:use

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()

    const onSearch = () => {
      viewRef.value?.search()
    }

    onMounted(async () => {
      onSearch()
    })
    watch(
      () => DATA.listStr,
      (n, o) => {
        if (n) DATA.list = n.split(',').map((i) => Number(i))
      }
    )
    watch(
      () => DATA.dansStr,
      (n, o) => {
        if (n) DATA.dans = n.split(',').map((i) => Number(i))
      }
    )
    const onCalc = () => {
      clc_svc(DATA.category, { list: DATA.list, dans: DATA.dans }).then((res) => {
        DATA.result = res.data.map((m) => m.join())
      })
    }
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <div>
          <ElInput type="textarea" v-model={DATA.listStr} row={3} />
          <ElSelect v-model={DATA.category}>
            <ElOption value={0}>旋转0</ElOption>
            <ElOption value={1}>旋转1</ElOption>
            <ElOption value={2}>旋转2</ElOption>
          </ElSelect>
          <ElInput type="textarea" v-model={DATA.dansStr} row={3} />
          <pre>{DATA.result.map((r) => r)}</pre>
          <ElButton onClick={onCalc}>计算</ElButton>
        </div>
      )
    }
    return rander
  } //end setup
})
