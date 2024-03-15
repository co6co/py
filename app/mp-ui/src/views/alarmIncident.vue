<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-select
            style="width: 160px"
            class="mr10"
            clearable
            v-model="table_module.query.siteId"
            placeholder="所属站点"
          >
            <el-option
              v-for="(item, index) in table_module.siteSelects"
              :key="index"
              :label="item.name"
              :value="item.id"
            />
          </el-select>

          <el-select
            style="width: 160px"
            class="mr10"
            clearable
            v-model="table_module.query.alarmType"
            placeholder="告警类型"
          >
            <el-option
              v-for="(item, index) in table_module.categoryList"
              :key="index"
              :label="item.desc"
              :value="item.alarmType"
            />
          </el-select>

          <el-date-picker
            style="margin-top: 3px"
            v-model="table_module.query.datetimes"
            format="YYYY-MM-DD HH:mm:ss"
            value-format="YYYY-MM-DD HH:mm:ss"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
            title="告警时间"
          />
          <el-link type="info" @click="setDatetime(0, 0.5)">30分钟内</el-link>
          <el-link type="info" @click="setDatetime(0, 1)">1小时内</el-link>
          <el-link type="info" @click="setDatetime(1, 24)">今天</el-link>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
        </div>
      </el-header>

      <el-main>
        <!--主内容-->
        <el-row>
          <el-col :span="11" style="height: 100%; overflow: auto">
            <el-scrollbar>
              <el-table
                highlight-current-row
                @sort-change="onColChange"
                :row-class-name="tableRowProp"
                :data="table_module.data"
                border
                class="table"
                ref="tableInstance"
                @row-click="onTableSelect"
                header-cell-class-name="table-header"
              >
                <el-table-column
                  label="序号"
                  width="119"
                  align="center"
                  :show-overflow-tooltip="true"
                >
                  <template #default="scope">
                    <span>{{ getTableIndex(table_module.query, scope.$index) }}</span>
                  </template>
                </el-table-column>
                <el-table-column
                  prop="siteName"
                  label="站点名"
                  width="119"
                  sortable
                  :show-overflow-tooltip="true"
                ></el-table-column>

                <el-table-column
                  prop="alarmTypeDesc"
                  label="告警类型"
                  width="119"
                  sortable
                  :show-overflow-tooltip="true"
                ></el-table-column>

                <el-table-column
                  prop="alarmTime"
                  label="告警时间"
                  sortable
                  :show-overflow-tooltip="true"
                ></el-table-column>
              </el-table>
            </el-scrollbar>
          </el-col>
          <el-col :span="13" style="position: relative">
            <div style="width: 100%; overflow: hidden; position: absolute; left: 0">
              <img-video :viewOption="form2.data"></img-video>
            </div>
          </el-col>
        </el-row>
      </el-main>
      <el-footer>
        <!--分页组件-->
        <div class="pagination">
          <el-pagination
            background
            layout="prev, pager,next,total,jumper"
            :current-page="table_module.query.pageIndex"
            :page-sizes="[100, 200, 300, 400]"
            :page-size="table_module.query.pageSize"
            :total="table_module.pageTotal"
            @current-change="onPageChange"
          >
          </el-pagination>
        </div>
      </el-footer>
    </el-container>

    <!-- 弹出框 -->
    <el-dialog
      title="详细信息"
      v-model="form2.dialogVisible"
      style="width: 98%; height: 90%"
      @keydown.ctrl="keyDown"
    >
      <el-row>
        <el-col :span="12">
          <img-video :viewOption="form2.data"></img-video>
        </el-col>
        <el-col :span="12"> </el-col>
      </el-row>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="form2.dialogVisible = false">取 消</el-button>
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
import {
  Delete,
  Edit,
  Search,
  Compass,
  MoreFilled,
  Download,
  ArrowUp,
  ArrowDown
} from '@element-plus/icons-vue'
import * as api from '../api/alarm'
import * as site_api from '../api/site'
import * as res_api from '../api'
import type { SelectItem } from '../api/types'
import { detailsInfo } from '../components/details'
import { imgVideo, types } from '../components/player'
import { type AlarmItem, getResources } from '../components/biz'
import { str2Obj, createStateEndDatetime } from '../utils'

import { showLoading, closeLoading } from '../components/Logining'
import { getTableIndex } from '../utils/tables'

interface AlertCategory {
  alarmType: string
  desc: string
}

interface Query extends IpageParam {
  datetimes: Array<string>
  alarmType?:string
  siteId?: number
}
interface table_module {
  query: Query
  moreOption: boolean
  data: AlarmItem[]
  currentRow?: AlarmItem
  pageTotal: number
  categoryList: AlertCategory[]
  siteSelects: SelectItem[]
}

const tableInstance = ref<InstanceType<typeof ElTable>>()
const currentTableItemIndex = ref<number>()
const table_module = reactive<table_module>({
  query: { 
    datetimes: [],
    pageIndex: 1,
    pageSize: 15,
    order: 'asc',
    orderBy: ''
  },
  moreOption: false,
  data: [],
  pageTotal: -1,
  categoryList: [],
  siteSelects: []
})

const setDatetime = (t: number, i: number) => {
  table_module.query.datetimes = createStateEndDatetime(t, i)
}
// 排序
const onColChange = (column: any) => {
  table_module.query.order = column.order === 'descending' ? 'desc' : 'asc'
  table_module.query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}

const tableRowProp = (data: { row: any; rowIndex: number }) => {
  data.row.index = data.rowIndex
  return ''
}

// 查询操作
const onSearch = () => {
  getData()
}
const getQuery = () => {
  table_module.query.pageIndex = table_module.query.pageIndex || 1
}
// 获取表格数据
const getData = () => {
  showLoading()
  getQuery()
  api
    .list_svc(table_module.query)
    .then((res) => {
      if (res.code == 0) {
        table_module.data = res.data
        table_module.pageTotal = res.total || -1
      } else {
        ElMessage.error(res.message)
      }
    })
    .finally(() => {
      closeLoading()
    })
}

//api 相关
const getAlarmCategory = async () => {
  const res = await api.alert_category_svc()
  if (res.code == 0) {
    table_module.categoryList = res.data
  }
  getData()
}
getAlarmCategory()

const getSiteSelectData = async () => {
  const res = await site_api.select_svc()
  if (res.code == 0) {
    table_module.siteSelects = res.data
  }
}
getSiteSelectData()

//end

// 分页导航
const onPageChange = (pageIndex: number) => {
  let totalPage = Math.ceil(table_module.pageTotal / table_module.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error('已经是第一页了')
  else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了')
  else (table_module.query.pageIndex = pageIndex), getData()
}
const setTableSelectItem = (index: number) => {
  if (
    tableInstance.value &&
    tableInstance.value.data &&
    index > -1 &&
    index < tableInstance.value.data.length
  ) {
    let row = tableInstance.value.data[index]
    tableInstance.value.setCurrentRow(row)
    onTableSelect(row)
  }
}
const onTableSelect = (row: any) => {
  currentTableItemIndex.value = row.index
  table_module.currentRow = row
  onOpen2Dialog(row)
}
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
      } else {
        if (v < 0) ElMessage.error('已经是第一条了')
        else if (tableInstance.value && v >= tableInstance.value.data.length)
          ElMessage.error('已经是最后一条了')
      }
    }
  }
  //process_view.value.keyDown(e)
  e.stopPropagation()
}

//**查看视频图片信息 */
interface dialog2DataType {
  dialogVisible: boolean
  data: Array<types.resourceOption>
}
let dialog2Data = {
  dialogVisible: false,
  data: []
}
let form2 = ref<dialog2DataType>(dialog2Data)
const getResultUrl = (uuid: string, isposter: boolean = false) => {
  if (isposter) return import.meta.env.VITE_BASE_URL + `/api/resource/poster/${uuid}/680/480`
  return import.meta.env.VITE_BASE_URL + `/api/resource/${uuid}`
}
const onOpen2Dialog = (row: AlarmItem) => {
  form2.value.data = getResources(row)
}
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
.el-row {
  height: 100%;
}
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

.formItem {
  display: flex;
  align-items: center;
  display: inline-block;
  .label {
    display: inline-block;
    color: #aaa;
    padding: 0 5px;
  }
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
