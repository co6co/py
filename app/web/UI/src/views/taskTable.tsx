import { defineComponent, nextTick, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElTag, ElButton, ElInput, ElTableColumn, ElMessage } from 'element-plus'
import { Search, Plus, Edit } from '@element-plus/icons-vue'

import { FormOperation, showLoading, closeLoading, IResponse } from 'co6co'
import {
  routeHook,
  DictSelect,
  DictSelectInstance,
  tableScope,
  TableView,
  TableViewInstance
} from 'co6co-right'

import { DictTypeCodes } from '@/api/app'
import Diaglog, { type Item } from '@/components/sys/modifyTask'

import { task_api as api } from '@/api/sys'
import ShowCode from '@/components/sys/showCode'

export default defineComponent({
  setup(prop, ctx) {
    //:define
    interface IQueryItem {
      category?: number
      state?: number
      name?: string
    }
    const DATA = reactive<{
      title?: string
      query: IQueryItem
      currentItem?: Item
      headItemWidth: { width: string }
    }>({
      query: {},
      headItemWidth: { width: '180px' }
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
    const statusDictRef = ref<DictSelectInstance>()

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
    const isRuning = (row: Item) => {
      const allStatusData = statusDictRef.value!.stateHook.selectData
      let result = false
      if (!allStatusData.value) return false
      allStatusData.value.forEach((item) => {
        if (item.flag == 'RUNNING' && row.execStatus.toString() == item.value) {
          result = true
          return true
        }
      })
      return result
    }

    const oncommand = (
      row: Item,
      svc: (id: number) => Promise<IResponse>,
      showCode: boolean = false
    ) => {
      showLoading()
      svc(row.id)
        .then((r) => {
          if (showCode) {
            if (typeof r.data == 'object') {
              r.data = JSON.stringify(r.data, null, 2)
            }
            showCodeRef.value?.openDialog({
              content: r.data,
              isPathon: false
            })
          } else {
            ElMessage.success(r.message)
            onRefesh()
          }
        })
        .finally(() => {
          closeLoading()
        })
    }
    const onSched = (row: Item) => {
      oncommand(row, api.exe_sched_svc)
    }
    const onStop = (row: Item) => {
      oncommand(row, api.stop_sched_svc)
    }
    const onOneRun = (row: Item) => {
      oncommand(row, api.exe_once_svc, true)
    }

    //:page reader
    const rander = (): VNodeChild => {
      return (
        <TableView dataApi={api.get_table_svc} ref={viewRef} query={DATA.query}>
          {{
            header: () => (
              <>
                <div class="handle-box">
                  <ElInput
                    style={DATA.headItemWidth}
                    clearable
                    v-model={DATA.query.name}
                    placeholder="模板标题"
                    class="handle-input"
                  />
                  <DictSelect
                    ref={categoryDictRef}
                    style={DATA.headItemWidth}
                    dictTypeCode={DictTypeCodes.TaskCategory}
                    v-model={DATA.query.category}
                    placeholder="类别"
                  />
                  <DictSelect
                    ref={stateDictRef}
                    style={DATA.headItemWidth}
                    dictTypeCode={DictTypeCodes.TaskState}
                    v-model={DATA.query.category}
                    placeholder="任务状态"
                  />
                  <DictSelect
                    ref={statusDictRef}
                    style={DATA.headItemWidth}
                    dictTypeCode={DictTypeCodes.TaskStatus}
                    v-model={DATA.query.category}
                    placeholder="运行状态"
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
                  label="cron[s]"
                  width="160"
                  prop="cron"
                  align="center"
                  sortable="custom"
                  showOverflowTooltip={true}
                />
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
                  label="运行状态"
                  prop="execStatus"
                  sortable="custom"
                  align="center"
                  width={200}
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTag>{statusDictRef.value?.getName(scope.row.execStatus)}</ElTag>
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
                <ElTableColumn label="操作" width={320} align="center" fixed="right">
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
                          disabled={isRuning(scope.row)}
                          onClick={() => onSched(scope.row)}
                        >
                          调度
                        </ElButton>
                        <ElButton
                          text={true}
                          disabled={!isRuning(scope.row)}
                          onClick={() => onStop(scope.row)}
                        >
                          停止
                        </ElButton>
                        <ElButton
                          text={true}
                          title="不要执行时间太长的程序"
                          showOverflowTooltip
                          onClick={() => onOneRun(scope.row)}
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
