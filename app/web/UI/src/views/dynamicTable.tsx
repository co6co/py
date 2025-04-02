import { defineComponent, nextTick, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElTag, ElButton, ElInput, ElTableColumn, ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Sugar, View, Edit } from '@element-plus/icons-vue'

import { FormOperation, showLoading, closeLoading } from 'co6co'
import {
  routeHook,
  DictSelect,
  DictSelectInstance,
  tableScope,
  TableView,
  TableViewInstance
} from 'co6co-right'

import { DictTypeCodes } from '@/api/app'
import Diaglog, { type Item } from '@/components/sys/modifyCode'
import ShowCode from '@/components/sys/showCode'

import { code_api as api } from '@/api/sys'

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
    const showCodeRef = ref<InstanceType<typeof ShowCode>>()
    const categoryDictRef = ref<DictSelectInstance>()
    const stateDictRef = ref<DictSelectInstance>()
    const isPythonCode = (category: number) => {
      const result = categoryDictRef.value?.flagIs(String(category), 'python')
      if (result == undefined) {
        return false
      }
      return result
    }
    const onOpenDialog = (row?: Item) => {
      DATA.title = row ? `编辑[${row?.name}]` : '增加'
      DATA.currentItem = row
      diaglogRef.value?.openDialog(row ? FormOperation.edit : FormOperation.add, row)
    }
    const onSearch = () => {
      //const targetRow = tableRef.value.bodyWrapper.querySelector(`tbody tr:nth-child(${index + 1})`)
      //console.info('123', viewRef.value?.tableRef?.$el)

      viewRef.value?.search()

      nextTick(() => {
        const index = 10
        // 获取目标行的 DOM 元素
        const targetRow = viewRef.value?.tableRef?.$el.querySelector(
          `tbody tr:nth-child(${index + 1})`
        )

        if (targetRow) {
          // 计算目标行相对于 el-scrollbar.wrap 的位置
          const targetOffsetTop = targetRow.offsetTop
          console.info(targetOffsetTop)
        }
      })
    }
    const onRefesh = () => {
      viewRef.value?.refesh()
    }
    onMounted(async () => {
      onSearch()
    })

    //:page reader
    const rander = (): VNodeChild => {
      return (
        <TableView dataApi={api.get_table_svc} ref={viewRef} query={DATA.query}>
          {{
            header: () => (
              <>
                <div class="handle-box">
                  <ElInput
                    width={160}
                    clearable
                    v-model={DATA.query.name}
                    placeholder="模板标题"
                    class="handle-input"
                  />
                  <DictSelect
                    ref={categoryDictRef}
                    width={160}
                    dictTypeCode={DictTypeCodes.CodeType}
                    v-model={DATA.query.category}
                    placeholder="类别"
                  />
                  <DictSelect
                    ref={stateDictRef}
                    width={160}
                    dictTypeCode={DictTypeCodes.CodeState}
                    v-model={DATA.query.category}
                    placeholder="状态"
                  />

                  <ElButton type="primary" icon={Search} onClick={onSearch}>
                    搜索
                  </ElButton>
                  <ElButton
                    type="primary"
                    icon={Plus}
                    v-permiss={getPermissKey(routeHook.ViewFeature.add)}
                    onClick={() => {
                      onOpenDialog()
                    }}
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
                  prop="code"
                  align="center"
                  width={180}
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="名称"
                  prop="name"
                  align="center"
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="类别"
                  prop="category"
                  sortable="custom"
                  align="center"
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: tableScope<Item>) => (
                      <>{categoryDictRef.value?.getName(scope.row.category)}</>
                    )
                  }}
                </ElTableColumn>

                <ElTableColumn
                  label="状态"
                  prop="state"
                  sortable="custom"
                  align="center"
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTag>{stateDictRef.value?.getName(scope.row.state)}</ElTag>
                    )
                  }}
                </ElTableColumn>

                <ElTableColumn
                  prop="createTime"
                  label="创建时间"
                  sortable="custom"
                  width={160}
                  show-overflow-tooltip={true}
                ></ElTableColumn>
                <ElTableColumn
                  prop="updateTime"
                  label="更新时间"
                  sortable="custom"
                  width={160}
                  show-overflow-tooltip={true}
                ></ElTableColumn>
                <ElTableColumn label="操作" width={220} align="center" fixed="right">
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
                          title="不要执行时间太长的程序"
                          disabled={!isPythonCode(scope.row.category)}
                          showOverflowTooltip
                          v-permiss={getPermissKey(routeHook.ViewFeature.check)}
                          onClick={() => {
                            showLoading()
                            api
                              .exe_once_svc(scope.row.id)
                              .then((r) => {
                                if (typeof r.data == 'object') {
                                  r.data = JSON.stringify(r.data, null, 2)
                                }
                                showCodeRef.value?.openDialog({
                                  content: r.data,
                                  isPathon: false
                                })
                              })
                              .finally(() => {
                                closeLoading()
                              })
                          }}
                        >
                          执行
                        </ElButton>
                      </>
                    )
                  }}
                </ElTableColumn>
              </>
            ),
            footer: () => (
              <>
                <Diaglog ref={diaglogRef} title={DATA.title} onSaved={onRefesh} />
                <ShowCode ref={showCodeRef} />
              </>
            )
          }}
        </TableView>
      )
    }
    return rander
  } //end setup
})
