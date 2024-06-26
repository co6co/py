<!--标签管理-->
<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <el-row :gutter="24">
          <div class="handle-box">
            <el-tree-select
              ref="elTreeInstance"
              v-model="selectedVal"
              class="handle-select mr10"
              lazy
              :load="onLoadTree"
              :props="tableModule.treeDataMap"
              :render-after-expand="false"
              show-checkbox
              :cache-data="cacheData"
              @check="Oncheck"
            />

            <el-input
              v-model="tableModule.query.boatName"
              style="width: 160px"
              placeholder="船名"
              class="handle-input mr10"
            />
            <el-select
              v-model="tableModule.query.flowStatus"
              style="width: 160px"
              class="mr10"
              placeholder="请选择状态"
            >
              <el-option
                v-for="item in formAttachData.flow_status"
                :key="item.key"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
            <el-select
              v-model="tableModule.query.manualAuditStatus"
              style="width: 160px"
              class="mr10"
              placeholder="人工审核状态"
            >
              <el-option
                v-for="item in formAttachData.manual_audit_state"
                :key="item.key"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
            <el-select
              v-model="tableModule.query.auditUser"
              class="handle-select mr10"
              style="width: 160px"
              placeholder="审核人员"
            >
              <el-option
                v-for="item in formAttachData.user_name_list"
                :key="item.key"
                :label="item.label"
                :value="item.value"
              />
            </el-select>

            <el-select
              v-model="tableModule.query.breakRules"
              class="handle-select mr10"
              style="width: 160px"
              placeholder="违反规则"
            >
              <el-option
                v-for="item in formAttachData.rule"
                :key="item.key"
                :label="item.value"
                :value="item.key"
              />
            </el-select>
            <el-link
              type="primary"
              title="更多"
              @click="tableModule.queryMoreOption = !tableModule.queryMoreOption"
              ><ElIcon :size="20"><MoreFilled /></ElIcon
            ></el-link>
            <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
            <el-button
              v-permiss="getPermissKey(ViewFeature.download)"
              type="danger"
              :icon="Download"
              @click="onDownload"
              >查询下载</el-button
            >
          </div>
        </el-row>
        <el-row v-if="tableModule.queryMoreOption" :gutter="24">
          <div class="handle-box">
            <div class="formItem">
              <span class="label" style="width: 160px">自动/人工审核：</span>
              <el-radio-group
                v-model="tableModule.query.auditStateEq"
                text-color="red"
                class="el-select mr10"
              >
                <el-checkbox
                  v-model="tableModule.query.includeAuditStateNull"
                  label="包含空"
                  :disabled="auditStateNullDisabled"
                />
                <el-radio :label="true">相同</el-radio>
                <el-radio :label="false">不同</el-radio>
                <el-radio>不启用</el-radio>
              </el-radio-group>
            </div>
          </div>
          <div class="handle-box">
            <div class="formItem">
              <span class="label" style="width: 80px">记录时间：</span>
              <el-link type="info" @click="setDatetime(0, 0.5)">0.5h内</el-link>
              <el-link type="info" @click="setDatetime(0, 1)">1h内</el-link>
              <el-link type="info" @click="setDatetime(1, 24)">今天</el-link>
              <el-date-picker
                v-model="tableModule.query.datetimes"
                style="margin-top: 3px"
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
          <el-table
            ref="tableInstance"
            highlight-current-row
            :row-class-name="tableRowProp"
            :data="tableModule.tableData"
            border
            class="table"
            header-cell-class-name="table-header"
            @sort-change="onColChange"
            @row-click="onTableSelect"
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
                  >{{ formAttachData.getFlowStateName(scope.row.flowStatus)?.label }}
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
                <el-tag :type="formAttachData.statue2TagType(scope.row.manualAuditResult)"
                  >{{ formAttachData.getManualStateName(scope.row.manualAuditResult)?.label }}
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
                ><el-tag :type="formAttachData.statue2TagType(scope.row.programAuditResult)">
                  {{ formAttachData.getAutoStateName(scope.row.programAuditResult)?.label }}
                </el-tag></template
              >
            </el-table-column>
            <el-table-column
              width="160"
              prop="devRecordTime"
              label="设备时间"
              sortable="custom"
              :show-overflow-tooltip="true"
            />
            <el-table-column label="操作" width="316" align="center">
              <template #default="scope">
                <el-button
                  v-permiss="getPermissKey(ViewFeature.associated)"
                  text
                  :icon="Edit"
                  @click="onOpenDialog(scope.row)"
                >
                  打标签
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-scrollbar>
      </el-main>
      <el-footer>
        <div class="pagination">
          <el-pagination
            background
            layout="prev, pager, next,total,jumper"
            :current-page="tableModule.query.pageIndex"
            :page-sizes="[100, 200, 300, 400]"
            :page-size="tableModule.query.pageSize"
            :total="tableModule.pageTotal"
            @current-change="onPageChange"
          />
        </div>
      </el-footer>
    </el-container>
  </div>

  <!-- 弹出框 -->
  <el-dialog
    v-model="form.dialogVisible"
    title="标注"
    style="width: 98%; height: 90%"
    @keydown.ctrl="keyDown"
  >
    <label-process
      ref="label_view"
      :options="tableModule.currentItem"
      :meta-data="formAttachData"
      @refesh="onRefesh"
    />
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="form.dialogVisible = false">取 消</el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts" name="basetable">
import {
  type PropType,
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
  watch
} from 'vue'
import {
  ElMessage,
  ElMessageBox,
  ElTable,
  ElTreeSelect,
  type FormInstance,
  type FormRules,
  dayjs
} from 'element-plus'
import { type TreeNode } from 'element-plus/es/components/tree-v2/src/types'
import { type TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import { Compass, Delete, Download, Edit, MoreFilled, Search } from '@element-plus/icons-vue'
import { createStateEndDatetime, closeLoading, showLoading } from 'co6co'
// eslint-disable-next-line camelcase
import { position_svc, queryList_svc, start_download_task } from '../api/process'

import { labelProcess } from '../components/labelprocess'
// eslint-disable-next-line camelcase
import { type table_module, type Item, types } from '../components/process'
import { form_attach_data as attachData } from '../store/process/viewdata'
import { ViewFeature, usePermission } from '../hook/sys/useRoute'

const { getPermissKey } = usePermission()
const formAttachData = reactive<ItemAattachData>(attachData)
const elTreeInstance = ref<InstanceType<typeof ElTreeSelect>>()

const selectedVal = ref<any[]>()
const cacheData = [{ value: 5, label: '位置信息' }]

const tableInstance = ref<InstanceType<typeof ElTable>>()
const currentTableItemIndex = ref<number>()

// eslint-disable-next-line camelcase
const tableModule = reactive<table_module>({
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
  return typeof tableModule.query.auditStateEq != 'boolean'
})

const setDatetime = (t: number, i: number) => {
  tableModule.query.datetimes = createStateEndDatetime(t, i)
}
// 排序
const onColChange = (column: any, prop: any, order: any) => {
  console.info('123', column, prop, order)
  tableModule.query.order = column.order === 'descending' ? 'desc' : 'asc'
  tableModule.query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}

const tableRowProp = (data: { row: any; rowIndex: number }) => {
  data.row.index = data.rowIndex
  return ''
}
const onRefesh = () => {
  tableModule.isResearch = false
  getData()
}

// 查询操作
const onSearch = () => {
  tableModule.isResearch = true
  getData()
}
watch(
  () => tableModule.tableData,
  async (newValue?: any, oldValue?: any) => {
    if (newValue) {
      let index = tableModule.isResearch ? 0 : currentTableItemIndex.value?.valueOf()
      if (index == undefined) index = 0
      await nextTick()
      setTableSelectItem(index)
    }
  }
)
const _setQueryContition = () => {
  tableModule.query.pageIndex = tableModule.query.pageIndex || 1
  if (elTreeInstance.value) {
    const allCheck: Array<{ groupType: string; serial: string; id: number }> =
      elTreeInstance.value.getCheckedNodes()
    tableModule.query.groupIds = allCheck
      .filter((m) => m.groupType == 'group' || m.groupType == 'company')
      .map((m) => m.id)
    tableModule.query.ipCameraSerial = allCheck
      .filter((m) => m.groupType == 'ch')
      .map((m) => m.serial)
    tableModule.query.boatSerials = allCheck
      .filter((m) => m.groupType == 'site')
      .map((m) => m.serial)
  }
}
// 获取表格数据
const getData = () => {
  _setQueryContition(),
    showLoading(),
    queryList_svc(tableModule.query)
      .then((res) => {
        if (res.code == 0) {
          tableModule.tableData = res.data
          tableModule.pageTotal = res.total || -1
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
  const totalPage = Math.ceil(tableModule.pageTotal / tableModule.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error('已经是第一页了')
  else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了')
  else (tableModule.query.pageIndex = pageIndex), getData()
}
const setTableSelectItem = (index: number) => {
  if (!tableInstance.value) return
  if (tableInstance.value.data && index > -1 && index < tableInstance.value.data.length) {
    const row = tableInstance.value.data[index]
    tableInstance.value.setCurrentRow(row)
    onTableSelect(row)
  }
}
const onTableSelect = (row: any) => {
  currentTableItemIndex.value = row.index
  tableModule.currentItem = row
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
// eslint-disable-next-line camelcase
const process_view = ref()
const keyDown = (e: KeyboardEvent) => {
  if (e.ctrlKey) {
    if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
      const current = tableModule.query.pageIndex.valueOf()
      const v = e.key == 'ArrowRight' || e.key == 'd' ? current + 1 : current - 1
      onPageChange(v)
    }
    if (tableInstance.value) {
      if (['ArrowUp', 'ArrowDown'].includes(e.key)) {
        let current = currentTableItemIndex.value
        if (!current) current = 0
        const v = e.key == 'ArrowDown' || e.key == 's' ? current + 1 : current - 1
        if (0 <= v && v < tableInstance.value.data.length) {
          setTableSelectItem(v)
        } else {
          if (v < 0) ElMessage.error('已经是第一条了')
          else if (v >= tableInstance.value.data.length) ElMessage.error('已经是最后一条了')
        }
      }
    }
  }
  // eslint-disable-next-line camelcase
  process_view.value.keyDown(e)
  e.stopPropagation()
}

const onDownload = () => {
  ElMessageBox.confirm(`确定以当前条件的创建下载任务吗？`, '提示', {
    type: 'warning'
  })
    .then(() => {
      _setQueryContition(),
        start_download_task(tableModule.query).then((res) => {
          if (res.code == 0) {
            ElMessageBox.alert(res.data, `"任务信息"${res.message}`)
          } else {
            ElMessageBox.alert(res.message, '任务创建失败')
          }
        })
    })
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    .catch(() => {})
}

//**打标签 */
const dialogData = {
  dialogVisible: false
}
const form = reactive(dialogData)
const onOpenDialog = (row?: any) => {
  form.dialogVisible = true
  tableModule.currentItem = row
}
//**end 打标签 */
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
