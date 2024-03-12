<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input
            v-model="table_module.query.name"
            placeholder="站点名称"
            class="handle-input mr10"
          ></el-input>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
          <el-button type="primary" :icon="Plus" @click="onOpenDialog(0)">新增</el-button>
        </div>
      </el-header>
      <el-main>
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
              prop="id"
              label="ID"
              width="80"
              align="center"
              sortable
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="name"
              label="名称"
              width="120"
              align="center"
              sortable
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column
              prop="deviceCode"
              label="设备代码"
              width="120"
              align="center"
              sortable
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column
              prop="deviceDesc"
              label="用途"
              width="120"
              align="center"
              sortable
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column
              prop="postionInfo"
              label="安装位置"
              width="120"
              align="center"
              sortable
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column
              width="160"
              prop="createTime"
              label="创建时间"
              sortable
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              width="160"
              prop="updateTime"
              label="更新时间"
              sortable
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column label="操作" align="center">
              <template #default="scope">
                <el-button text :icon="Edit" @click="onOpenDialog(1, scope.row)"> 修改 </el-button>
                <el-button text :icon="Setting" @click="onOpenConfigPage('router', scope.row)">
                  路由器
                </el-button>
                <el-button text :icon="Setting" @click="onOpenConfigPage('box', scope.row)">
                  AI盒子
                </el-button>
                <el-button text :icon="Setting" @click="onOpenConfigPage('ip_camera', scope.row)">
                  监控球机
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-scrollbar>
      </el-main>
      <el-footer>
        <div class="pagination">
          <!--prev, pager, next,total,jumper-->
          <el-pagination
            background
            :current-page="table_module.query.pageIndex"
            :page-sizes="[10, 20, 30, 100, 200, 300, 1000]"
            layout="prev, pager, next,total,jumper"
            @update:page-size="onPageSize"
            :page-size="table_module.query.pageSize"
            :total="table_module.pageTotal"
            prev-text="上一页"
            next-text="下一页"
            @current-change="onPageChange"
          >
          </el-pagination>
        </div>
      </el-footer>
    </el-container>

    <!-- 弹出框 -->
    <el-dialog
      title="打开配置网址"
      draggable
      class="setting"
      v-model="configDialog_model.showDialog"
      style="width: 36%"
    >
      <template #header>
        <el-text>
          <el-icon>
            <Setting />
          </el-icon>
          打开配置网址
        </el-text>
      </template>
      <div>
        <el-button type="primary" size="small" :icon="Setting" @click="openBlankPage(0)"
          >内网配置</el-button
        >
        <el-button type="primary" size="small" :icon="Setting" @click="openBlankPage(1)"
          >外网配置</el-button
        >
        <el-button
          type="primary"
          size="small"
          v-if="configDialog_model.deviceData && configDialog_model.deviceData.sshConfigUrl"
          :icon="Setting"
          @click="openBlankPage(2)"
          >SSH配置</el-button
        >
      </div>
    </el-dialog>

    <!-- 弹出框 -->
    <edit-box ref="editBoxDialogRef" :label-width="90" @saved="getData"  :model="form.fromData" :title="form.title" style="width:40%;height:56%;"></edit-box> 
  </div>
</template>

<script setup lang="ts" name="basetable">
import {
  ref,
  watch,
  markRaw,
  reactive,
  nextTick,
  type PropType,
  onMounted,
  onBeforeUnmount,
  computed
} from 'vue'
import {
  ElMessage,
  ElForm,
  ElMessageBox,
  type FormRules,
  type FormInstance,
  ElTreeSelect,
  dayjs
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
  Plus,
  Minus,
  Message,
  Cpu,
  VideoCamera,
  Setting
} from '@element-plus/icons-vue'
import * as api from '../api/site'
import * as dev_api from '../api/device'
import * as res_api from '../api'
import * as t from '../store/types/devices'
import { detailsInfo } from '../components/details'
import { imgVideo, types } from '../components/player'
import { str2Obj, createStateEndDatetime } from '../utils'
import { showLoading, closeLoading } from '../components/Logining'
import { pagedOption, type PagedOption } from '../components/tableEx'

import editBox,{ type Item} from '../components/biz/src/editBox'

 
 
interface Query extends IpageParam {
  name: string
  category?: number
  datetimes: Array<string>
}
interface table_module {
  query: Query
  moreOption: boolean
  data: Item[]
  currentRow?: Item
  pageTotal: number
}

const tableInstance = ref<any>(null)
const currentTableItemIndex = ref<number>()
const table_module = reactive<table_module>({
  query: {
    name: '',
    datetimes: [],
    pageIndex: 1,
    pageSize: 10,
    order: 'asc',
    orderBy: ''
  },
  moreOption: false,
  data: [],
  pageTotal: -1
})

// 排序
const onColChange = (column: any) => {
  table_module.query.order = column.order === 'descending' ? 'desc' : 'asc'
  table_module.query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}
const tableRowProp = (data: { row: any; rowIndex: number }) => {
  data.row.index = data.rowIndex
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
  getQuery(),
    api
      .list2_svc(table_module.query)
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
getData()
const onPageSize = (size: number) => {
  table_module.query.pageSize = size
}
// 分页导航
const onPageChange = (pageIndex: number) => {
  let totalPage = Math.ceil(table_module.pageTotal / table_module.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error('已经是第一页了')
  else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了')
  else (table_module.query.pageIndex = pageIndex), getData()
}

const onTableSelect = (row: any) => {
  currentTableItemIndex.value = row.index
  table_module.currentRow = row
}

//**详细信息 */
 
interface dialogDataType {
  dialogVisible: boolean
  operation: 0 | 1 | number
  title?: string
  id: number
  fromData: Item
}
let dialogData = {
  dialogVisible: false,
  operation: 0,
  id: 0,
  fromData: {
    id:-1,
    name: '',
    postionInfo: '',
    deviceCode: '',
    deviceDesc: '',
    createTime: "",
  updateTime: ""
  }
} 
const editBoxDialogRef=ref<InstanceType <typeof editBox>>()
let form = reactive<dialogDataType>(dialogData)
const onOpenDialog = (operation: 0 | 1, row?: Item) => { 
  table_module.currentRow = row
  switch (operation) {
    case 0:
      form.title = '增加' 
      break
    case 1: 
      form.title = '编辑' 
      break
  } 
  editBoxDialogRef.value?.openDialog(operation,row) 
}
 
const openBlankPage = (type: number) => {
  if (!configDialog_model.deviceData) {
    ElMessage.warning('未关联该设备数据！')
    return
  }
  console.info(type, configDialog_model.deviceData)
  switch (type) {
    case 0:
      window.open(configDialog_model.deviceData.innerConfigUrl, '_blank')
    case 1:
      window.open(configDialog_model.deviceData.configUrl, '_blank')
      break
    case 2:
      window.open(configDialog_model.deviceData.sshConfigUrl, '_blank')
      break
    default:
      window.open(configDialog_model.deviceData.configUrl, '_blank')
  }
}

const configDialog_model = reactive<{
  showDialog: boolean
  deviceData?: { configUrl: string; innerConfigUrl: string; sshConfigUrl?: string }
}>({
  showDialog: false
}) 
//打开配置页面
const onOpenConfigPage = (category: string, row: Item) => {
  api.getDetailInfo(row.id, category).then((res) => {
    if (res.code == 0) {
      let data = []
      for (let i = 0; i < res.data.length; i++) {
        data.push({ name: res.data[i].name + '信息', data: res.data[i] })
      }
      if (res.data.length == 0) ElMessage.warning('未关联该设备！')
      else (configDialog_model.showDialog = true), (configDialog_model.deviceData = res.data[0])
    }
  })
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

::v-deep .streamInfo {
  max-width: 560px;

  .el-card {
    margin: 2px 0;
  }

  .el-form-item {
    padding: 8px 0;
  }

  .el-form-item__content {
    width: 470px;
  }

  .el-card__body {
    padding: 2px 5px;
  }

  .el-form-item__label {
    width: 70px;
  }
}

::v-deep .setting {
  .el-dialog__body {
    padding: 10px 40px 20px 40px;
  }
}
</style>
