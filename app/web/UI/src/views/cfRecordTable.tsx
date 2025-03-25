import { defineComponent, nextTick, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElButton, ElInput, ElTableColumn, ElMessage } from 'element-plus'
import { Search, Plus, Edit, Delete } from '@element-plus/icons-vue'

import { FormOperation } from 'co6co'
import {
  routeHook,
  DictSelectInstance,
  tableScope,
  TableView,
  TableViewInstance,
  deleteHook
} from 'co6co-right'

import * as api from '@/api/transimit/cf'
import { type IListItem as Item } from '@/api/transimit/cf'
import Diaglog from '@/components/transmit/modifyRecord'

export default defineComponent({
  setup(prop, ctx) {
    //:define
    interface IQueryItem {
      category?: number
      state?: number
      name?: string
    }
    const DATA = reactive<{ title?: string; query: IQueryItem; currentItem?: Item }>({
      query: {}
    })

    //:use
    const { getPermissKey } = routeHook.usePermission()

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()
    const diaglogRef = ref<InstanceType<typeof Diaglog>>()

    const onOpenDialog = (row?: api.IListItem) => {
      try {
        DATA.title = row ? `编辑[${row?.name}]` : '增加'
        DATA.currentItem = row

        diaglogRef.value?.openDialog(row ? FormOperation.edit : FormOperation.add, DATA.currentItem)
      } catch (e) {
        ElMessage.error('打开对话框失败' + e)
      }
    }
    const onSearch = () => {
      viewRef.value?.search()
    }
    const onRefesh = () => {
      viewRef.value?.refesh()
    }
    const onFilter = (data: api.IListResult) => {
      return data.result
    }
    onMounted(async () => {
      onSearch()
    })
    const { deleteSvc } = deleteHook.default(api.delete_svc, () => {
      onRefesh()
    })
    const onDelete = (row: Item) => {
      deleteSvc(row.id, row.name)
    }

    //:page reader
    const rander = (): VNodeChild => {
      return (
        <TableView
          dataApi={api.list_svc}
          showPaged={false}
          ref={viewRef}
          resultFilter={onFilter}
          query={DATA.query}
        >
          {{
            header: () => (
              <>
                <div class="handle-box">
                  <ElInput
                    style="width: 160px"
                    clearable
                    v-model={DATA.query.name}
                    placeholder="模板标题"
                    class="handle-input"
                  />

                  <ElButton type="primary" icon={Search} onClick={onSearch}>
                    搜索
                  </ElButton>
                  <ElButton
                    type="primary"
                    icon={Plus}
                    v-permiss={getPermissKey(routeHook.ViewFeature.add)}
                    onClick={() => onOpenDialog()}
                  >
                    新增
                  </ElButton>
                </div>
              </>
            ),
            default: () => (
              <>
                <ElTableColumn label="序号" width={55} align="center">
                  {{
                    default: (scope: tableScope) => viewRef.value?.rowIndex(scope.$index)
                  }}
                </ElTableColumn>
                <ElTableColumn
                  label="编号"
                  prop="id"
                  align="center"
                  width={180}
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="名称"
                  prop="name"
                  align="left"
                  width={180}
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="类型"
                  prop="type"
                  width={80}
                  align="center"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="记录值"
                  prop="content"
                  width={200}
                  align="left"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="可代理"
                  prop="proxiable"
                  align="center"
                  width={80}
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="代理"
                  prop="proxied"
                  align="center"
                  width={80}
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="ttl"
                  prop="ttl"
                  align="center"
                  sortable="custom"
                  width={80}
                  showOverflowTooltip={true}
                />

                <ElTableColumn
                  prop="created_on"
                  label="创建时间"
                  width={160}
                  show-overflow-tooltip={true}
                />
                <ElTableColumn
                  prop="modified_on"
                  label="更新时间"
                  width={160}
                  show-overflow-tooltip={true}
                />
                <ElTableColumn label="操作" width={260} align="center" fixed="right">
                  {{
                    default: (scope: tableScope<Item>) => (
                      <>
                        <ElButton
                          text={true}
                          icon={Edit}
                          onClick={() => onOpenDialog(scope.row)}
                          v-permiss={getPermissKey(routeHook.ViewFeature.edit)}
                        >
                          编辑
                        </ElButton>
                        <ElButton
                          text={true}
                          icon={Delete}
                          onClick={() => onDelete(scope.row)}
                          v-permiss={getPermissKey(routeHook.ViewFeature.del)}
                        >
                          删除
                        </ElButton>
                      </>
                    )
                  }}
                </ElTableColumn>
              </>
            ),
            footer: () => (
              <>
                <Diaglog ref={diaglogRef} title={DATA.title} onSaved={onRefesh}></Diaglog>
              </>
            )
          }}
        </TableView>
      )
    }
    return rander
  } //end setup
})
