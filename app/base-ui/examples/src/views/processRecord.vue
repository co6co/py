<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <el-row :gutter="24">
          <div class="handle-box">
            <el-tree-select
              class="handle-select mr10"
              ref="elTreeInstance"
              v-model="selectedVal"
              lazy
              :load="onLoadTree"
              :props="table_module.treeDataMap"
              :render-after-expand="false"
              show-checkbox
              @check="Oncheck"
              :cache-data="cacheData"
            />
            <el-input
              style="width: 160px"
              v-model="table_module.query.id"
              placeholder="记录ID"
              class="handle-input mr10"
            />
            <el-input
              style="width: 160px"
              v-model="table_module.query.boatName"
              placeholder="船名"
              class="handle-input mr10"
            />

            <el-select
              style="width: 160px"
              class="mr10"
              v-model="table_module.query.flowStatus"
              placeholder="请选择"
            >
              <el-option
                v-for="item in form_attach_data.flow_status"
                :key="item.key"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
            <el-select
              style="width: 160px"
              class="mr10"
              v-model="table_module.query.manualAuditStatus"
              placeholder="请选择"
            >
              <el-option
                v-for="item in form_attach_data.manual_audit_state"
                :key="item.key"
                :label="item.label"
                :value="item.value"
              />
            </el-select>

            <el-select
              style="width: 160px"
              class="mr10"
              v-model="table_module.query.auditState"
              placeholder="请选择"
            >
              <el-option
                v-for="item in form_attach_data.ai_audit_state"
                :key="item.key"
                :label="item.label"
                :value="item.value"
              />
            </el-select>

            <el-link
              type="primary"
              title="更多"
              @click="table_module.queryMoreOption = !table_module.queryMoreOption"
              ><ElIcon :size="20"><MoreFilled /></ElIcon
            ></el-link>
            <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
            <el-button
              type="danger"
              v-permiss="getPermissKey(ViewFeature.downloads)"
              :icon="Download"
              @click="onDownload"
              >查询下载</el-button
            >
          </div>
        </el-row>
        <el-row :gutter="24" v-if="table_module.queryMoreOption">
          <div class="handle-box">
            <div class="formItem">
              <span class="label" style="width: 160px">自动/人工审核：</span>
              <el-radio-group
                v-model="table_module.query.auditStateEq"
                text-color="red"
                class="el-select mr10"
              >
                <el-checkbox
                  v-model="table_module.query.includeAuditStateNull"
                  label="包含空"
                  :disabled="auditStateNullDisabled"
                />
                <el-radio :label="true">相同</el-radio>
                <el-radio :label="false">不同</el-radio>
                <el-radio>不启用</el-radio>
              </el-radio-group>
            </div>

            <el-select
              clearable
              style="width: 160px"
              class="mr10"
              v-model="table_module.query.auditUser"
              placeholder="审核人员"
            >
              <el-option
                v-for="item in form_attach_data.user_name_list"
                :key="item.key"
                :label="item.label"
                :value="item.value"
              />
            </el-select>

            <el-select
              style="width: 160px"
              class="mr10"
              v-model="table_module.query.breakRules"
              placeholder="违反规则"
            >
              <el-option
                v-for="item in form_attach_data.rule"
                :key="item.key"
                :label="item.value"
                :value="item.key"
              />
            </el-select>
          </div>
          <div class="handle-box">
            <div class="formItem">
              <span class="label" style="width: 80px">记录时间：</span>
              <el-link type="info" @click="setDatetime(0, 0.5)">0.5h内</el-link>
              <el-link type="info" @click="setDatetime(0, 1)">1h内</el-link>
              <el-link type="info" @click="setDatetime(1, 24)">今天</el-link>
              <el-date-picker
                style="margin-top: 3px"
                v-model="table_module.query.datetimes"
                format="YYYY-MM-DD HH:mm:ss"
                value-format="YYYY-MM-DD HH:mm:ss"
                type="datetimerange"
                range-separator="至"
                start-placeholder="开始时间"
                end-placeholder="结束时间"
                title="设备时间"
              />
            </div>
          </div>
        </el-row>
      </el-header>
      <el-main>
        <el-scrollbar>
          <el-row @keydown.ctrl="keyDown" @keydown.prevent="keyDown">
            <el-col :span="14">
              <el-table
                highlight-current-row
                @sort-change="onColChange"
                @row-contextmenu="onRowContext"
                :row-class-name="tableRowProp"
                :data="table_module.tableData"
                border
                class="table"
                ref="tableInstance"
                @row-click="onTableSelect"
                header-cell-class-name="table-header"
              >
                <el-table-column prop="id" label="ID" width="90" align="center" sortable="custom" />
                <el-table-column
                  prop="boatName"
                  label="船名"
                  width="90"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                />
                <el-table-column
                  prop="vioName"
                  label="违规名称"
                  width="110"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                />
                <el-table-column label="处理状态" width="110" sortable="custom" prop="flowStatus">
                  <template #default="scope">
                    <el-tag
                      >{{ form_attach_data.getFlowStateName(scope.row.flowStatus)?.label }}
                    </el-tag></template
                  >
                </el-table-column>
                <el-table-column
                  label="人工审核"
                  width="120"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                  prop="manualAuditResult"
                >
                  <template #default="scope">
                    <el-tag :type="form_attach_data.statue2TagType(scope.row.manualAuditResult)"
                      >{{ form_attach_data.getManualStateName(scope.row.manualAuditResult)?.label }}
                    </el-tag></template
                  >
                </el-table-column>
                <el-table-column
                  label="程序审核"
                  width="120"
                  sortable="custom"
                  prop="programAuditResult"
                  :show-overflow-tooltip="true"
                >
                  <template #default="scope"
                    ><el-tag :type="form_attach_data.statue2TagType(scope.row.programAuditResult)">
                      {{ form_attach_data.getAutoStateName(scope.row.programAuditResult)?.label }}
                    </el-tag></template
                  >
                </el-table-column>
                <el-table-column
                  width="160"
                  prop="devRecordTime"
                  label="设备时间"
                  sortable="custom"
                  :sort-method="dateOrderBy"
                  :show-overflow-tooltip="true"
                />
                <el-table-column
                  width="160"
                  prop="webRecordTime"
                  label="平台时间"
                  sortable="custom"
                  :sort-method="dateOrderBy"
                  :show-overflow-tooltip="true"
                />
                <el-table-column
                  width="160"
                  prop="auditDownloadTime"
                  label="审核下载"
                  sortable="custom"
                  :sort-method="dateOrderBy"
                  :show-overflow-tooltip="true"
                />
                <el-table-column
                  width="160"
                  prop="auditInferTime"
                  label="审核推理"
                  sortable="custom"
                  :sort-method="dateOrderBy"
                  :show-overflow-tooltip="true"
                />
                <el-table-column
                  width="160"
                  prop="auditAutoPushTime"
                  label="自动推送"
                  sortable="custom"
                  :sort-method="dateOrderBy"
                  :show-overflow-tooltip="true"
                />
                <el-table-column
                  width="160"
                  prop="manualAuditTime"
                  label="人工审核"
                  sortable="custom"
                  :sort-method="dateOrderBy"
                  :show-overflow-tooltip="true"
                />
                <el-table-column
                  width="160"
                  prop="manualPushTime"
                  label="人工下发"
                  sortable="custom"
                  :sort-method="dateOrderBy"
                  :show-overflow-tooltip="true"
                />
              </el-table>
              <div class="pagination">
                <el-pagination
                  background
                  layout="prev, pager, next,total,jumper"
                  :current-page="table_module.query.pageIndex"
                  :page-sizes="[100, 200, 300, 400]"
                  :page-size="table_module.query.pageSize"
                  :total="table_module.pageTotal"
                  @current-change="onPageChange"
                />
              </div>
            </el-col>
            <el-col :span="10">
              <process-data-view
                style="padding-left: 12px"
                ref="process_view"
                :options="table_module.currentItem"
                :meta-data="form_attach_data"
                @refesh="onRefesh"
              />
            </el-col>
          </el-row>
        </el-scrollbar>
      </el-main>
    </el-container>
    <!--表格右键菜单-->
    <div
      class="menuInfo"
      v-if="table_menuInfo.visible"
      :style="{ left: table_menuInfo.left + 'px', top: table_menuInfo.top + 'px' }"
      style="position: fixed; z-index: 9"
    >
      <el-menu mode="vertical" @select="onSelectMenu">
        <el-menu-item v-permiss="getPermissKey(ViewFeature.view)" index="1">查看日志</el-menu-item>
      </el-menu>
    </div>

    <!-- 弹出框 -->
    <el-dialog title="日志信息" v-model="showLog" width="90%" style="height: 90%; overflow: hidden">
      <log :title="LogData.title" :content="LogData.content" />
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showLog = false">关 闭</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts" name="basetable">
import {
  ref,
  watch,
  reactive,
  nextTick,
  type PropType,
  onMounted,
  onBeforeUnmount,
  computed
} from 'vue'

import {
  ElMessage,
  ElMessageBox,
  type FormRules,
  type FormInstance,
  ElTreeSelect,
  dayjs,
  ElTable
} from 'element-plus'
import { type TreeNode } from 'element-plus/es/components/tree-v2/src/types'
import { type TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import { Delete, Edit, Search, Compass, MoreFilled, Download } from '@element-plus/icons-vue'
import {
  // eslint-disable-next-line camelcase
  queryList_svc,
  // eslint-disable-next-line camelcase
  position_svc,
  // eslint-disable-next-line camelcase
  start_download_task,
  // eslint-disable-next-line camelcase
  get_log_content_svc
} from '../api/process'
import {
  processDataView,
  tree,
  selectTree,
  types,
  type table_module,
  type Item
} from '../components/process'
import { log } from '../components/log'
import { form_attach_data as attach_data } from '../store/process/viewdata'
import { createStateEndDatetime } from 'co6co'
import { showLoading, closeLoading } from '../components/Logining'
import { usePermission, ViewFeature } from '../hook/sys/useRoute'
const { getPermissKey } = usePermission()

let form_attach_data = reactive<ItemAattachData>(attach_data)
const elTreeInstance = ref<InstanceType<typeof ElTreeSelect>>()
const selectedVal = ref<any[]>()
const cacheData = [{ value: 5, label: '位置信息' }]

const tableInstance = ref<InstanceType<typeof ElTable>>()
const currentTableItemIndex = ref<number>()
const table_menuInfo = reactive<{ visible: boolean; left: number; top: number; row: any }>({
  visible: false,
  left: 0,
  top: 0,
  row: {}
})

const onRowContext = (row: any, column: any, event: any) => {
  event.preventDefault() //阻止鼠标右键默认行为
  table_menuInfo.row = row
  table_menuInfo.left = event.clientX
  table_menuInfo.top = event.pageY
  table_menuInfo.visible = true
}
const showLog = ref(false)
const LogData = ref<{ title: string; content: string }>({ title: '', content: '' })
const onSelectMenu = async () => {
  const id = table_menuInfo.row?.id
  table_menuInfo.visible = false
  if (id) {
    if (table_menuInfo.row.auditLogPath)
      (LogData.value.title = table_menuInfo.row.auditLogPath),
        get_log_content_svc(id)
          .then((res) => {
            LogData.value.content = res.data
          })
          .catch((err) => ((LogData.value.content = err.response.statusText), console.error(err)))
    else (LogData.value.title = '日志文件不存在'), (LogData.value.content = '')
  }
  showLog.value = true
}
const table_module = reactive<table_module>({
  isResearch: true,
  query: {
    boatName: '',
    flowStatus: '',
    manualAuditStatus: '',
    auditState: '',
    datetimes: [],
    includeAuditStateNull: false,
    auditStateEq: undefined,
    breakRules: '',
    pageIndex: 1,
    pageSize: 15,
    order: 'asc',
    orderBy: '',
    groupIds: [],
    boatSerials: [],
    ipCameraSerial: []
  },
  queryMoreOption: false,
  cache: {},
  tableData: [],
  pageTotal: -1,
  //树形选择
  treeData: [],
  treeCacheData: [{ value: 5, label: '位置信息' }],
  treeDataMap: {
    value: 'id',
    label: 'name',
    children: 'children',
    isLeaf: 'isLeaf'
  }
})
const auditStateNullDisabled = computed(() => {
  return typeof table_module.query.auditStateEq != 'boolean'
})
const setDatetime = (t: number, i: number) => {
  table_module.query.datetimes = createStateEndDatetime(t, i)
}
// 排序
const onColChange = (column: any, prop: any, order: any) => {
  table_module.query.order = column.order === 'descending' ? 'desc' : 'asc'
  table_module.query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}

const dateOrderBy = (a: any, b: any) => {}

const tableRowProp = (data: { row: any; rowIndex: number }) => {
  data.row.index = data.rowIndex
  return ''
}
const onRefesh = () => {
  table_module.isResearch = false
  getData()
}

// 查询操作
const onSearch = () => {
  table_module.isResearch = true
  getData()
}
watch(
  () => table_module.tableData,
  async (newValue?: any, oldValue?: any) => {
    //console.info(table_module.isResearch,newValue)
    if (newValue) {
      let index = table_module.isResearch ? 0 : currentTableItemIndex.value?.valueOf()
      if (index == undefined) index = 0
      await nextTick()
      setTableSelectItem(index)
    }
  }
)
const _setQueryContition = () => {
  table_module.query.pageIndex = table_module.query.pageIndex || 1
  if (elTreeInstance.value) {
    let allCheck: Array<{ groupType: string; serial: string; id: number }> =
      elTreeInstance.value.getCheckedNodes()
    table_module.query.groupIds = allCheck
      .filter((m) => m.groupType == 'group' || m.groupType == 'company')
      .map((m) => m.id)
    table_module.query.ipCameraSerial = allCheck
      .filter((m) => m.groupType == 'ch')
      .map((m) => m.serial)
    table_module.query.boatSerials = allCheck
      .filter((m) => m.groupType == 'site')
      .map((m) => m.serial)
  }
}
// 获取表格数据
const getData = () => {
  _setQueryContition(),
    showLoading(),
    queryList_svc(table_module.query)
      .then((res) => {
        if (res.code == 0) {
          table_module.tableData = res.data
          table_module.pageTotal = res.total || -1
        } else {
          ElMessage.error(res.message)
        }
      })
      .finally(() => {
        closeLoading()
      })
}
getData()
// 分页导航
const onPageChange = (pageIndex: number) => {
  let totalPage = Math.ceil(table_module.pageTotal / table_module.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error('已经是第一页了')
  else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了')
  else (table_module.query.pageIndex = pageIndex), getData()
}
const setTableSelectItem = (index: number) => {
  if (!tableInstance.value) return
  if (tableInstance.value.data && index > -1 && index < tableInstance.value.data.length) {
    let row = tableInstance.value.data[index]
    tableInstance.value.setCurrentRow(row)
    onTableSelect(row)
  }
}
const onTableSelect = (row: Item) => {
  currentTableItemIndex.value = row.index
  table_module.currentItem = row
}

//tree
const onLoadTree = (node: TreeNodeData, resolve: any) => {
  if (node.isLeaf) return resolve([])
  position_svc({ pid: node.data.id }).then((res) => {
    if (res.code == 0) {
      res.data.forEach((e: { groupType: string; isLeaf: boolean }) => {
        //console.info(e.groupType)
        if (e.groupType == 'ch') e.isLeaf = true
        else e.isLeaf = false
      })
      resolve(res.data)
    }
  })
}

const Oncheck = (
  data: TreeNodeData,
  node: { checkedNodes: Array<{ id: number; name: string }> }
) => {
  selectedVal.value = node.checkedNodes.map((m) => m.name)
}
const process_view = ref()
const keyDown = (e: KeyboardEvent) => {
  if (e.ctrlKey) {
    if (['ArrowLeft', 'ArrowRight'].indexOf(e.key) > -1) {
      let current = table_module.query.pageIndex.valueOf()
      let v = e.key == 'ArrowRight' || e.key == 'd' ? current + 1 : current - 1
      onPageChange(v)
    }
    if (['ArrowUp', 'ArrowDown'].indexOf(e.key) > -1) {
      let current = currentTableItemIndex.value
      if (!current) current = 0
      let v = e.key == 'ArrowDown' || e.key == 's' ? current + 1 : current - 1

      if (tableInstance.value && 0 <= v && v < tableInstance.value.data.length) {
        setTableSelectItem(v)
      } else if (tableInstance.value) {
        if (v < 0) ElMessage.error('已经是第一条了')
        else if (v >= tableInstance.value.data.length) ElMessage.error('已经是最后一条了')
      }
    }
  }
  process_view.value.keyDown(e)
  e.stopPropagation()
}

const onDownload = () => {
  ElMessageBox.confirm(`确定以当前条件的创建下载任务吗？`, '提示', {
    type: 'warning'
  })
    .then(() => {
      _setQueryContition(),
        start_download_task(table_module.query).then((res) => {
          if (res.code == 0) {
            ElMessageBox.alert(res.data, `"任务信息"${res.message}`)
          } else {
            ElMessageBox.alert(res.message, '任务创建失败')
          }
        })
    })
    .catch(() => {})
}
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';

.view .title {
  color: var(--el-text-color-regular);
  font-size: 18px;
  margin: 10px 0;
}
.view .value {
  color: var(--el-text-color-primary);
  font-size: 16px;
  margin: 10px 0;
}

::v-deep .view .radius {
  height: 40px;
  width: 70%;
  border: 1px solid var(--el-border-color);
  border-radius: 0;
  margin-top: 20px;
}
::v-deep .el-table tr,
.el-table__row {
  cursor: pointer;
}
::v-deep .el-dialog__body {
  height: 70%;
  overflow: auto;
}
.menuInfo {
  .el-menu {
    width: auto;
    .el-menu-item {
      padding: 10px;
      height: 40px;
    }
  }
}
</style>
