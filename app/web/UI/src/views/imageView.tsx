import { defineComponent, nextTick, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import {
  ElTag,
  ElButton,
  ElInput,
  ElTableColumn,
  ElMessage,
  ElSelect,
  ElOption,
  ElImage
} from 'element-plus'
import { Search, Plus, Edit } from '@element-plus/icons-vue'

import { getBaseUrl } from 'co6co'
import { TableView, TableViewInstance, ImageView, image2Option } from 'co6co-right'

import * as api from '@/api/dev'
export default defineComponent({
  setup(prop, ctx) {
    //:define
    interface IQueryItem {
      date?: string
    }
    const DATA = reactive<{
      title?: string
      query: IQueryItem
      headItemWidth: { width: string }
      selectData: Array<string>
    }>({
      query: {},
      headItemWidth: { width: '180px' },
      selectData: []
    })

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()

    const onSearch = () => {
      viewRef.value?.search()
    }

    onMounted(async () => {
      api.img_folder_select_svc().then((res) => {
        DATA.selectData = res.data
        if (res.data.length > 0) DATA.query.date = DATA.selectData[0]
        onSearch()
      })
    })
    const onFilter = (data: Array<string>) => {
      //const result: Array<image2Option> = []
      const result = data.map((v) => {
        const item = {
          url: `${getBaseUrl()}/api/dev/img/preview/${DATA.query.date}/${v}`,
          authon: true
        }
        return { option: item }
      })
      return result
    }
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <TableView
          dataApi={api.img_list_svc}
          resultFilter={onFilter}
          ref={viewRef}
          query={DATA.query}
        >
          {{
            header: () => (
              <>
                <div class="handle-box">
                  <ElSelect style={DATA.headItemWidth} v-model={DATA.query.date}>
                    {DATA.selectData.map((v, i) => (
                      <ElOption label={v} key={i} value={v}></ElOption>
                    ))}
                  </ElSelect>

                  <ElButton type="primary" icon={Search} onClick={onSearch}>
                    搜索
                  </ElButton>
                </div>
              </>
            ),
            default: () => (
              <>
                <ElTableColumn
                  label="编号"
                  prop="code"
                  align="center"
                  width={180}
                  sortable="custom"
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: { option: image2Option } }) => (
                      <ImageView option={scope.row.option}></ImageView>
                    )
                  }}
                </ElTableColumn>
              </>
            ),
            footer: () => <></>
          }}
        </TableView>
      )
    }
    return rander
  } //end setup
})
