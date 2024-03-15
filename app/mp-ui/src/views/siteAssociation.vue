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

            <el-table-column label="路由器" align="center">
              <template #default="scope">
                <el-button text :icon="Edit" @click="onOpenRouterDialog(scope.row)"> 修改 </el-button>
                <el-button text :icon="Message" @click="onOpen2Dialog('router', scope.row)">
                  路由器
                </el-button>
              </template>
            </el-table-column>

            <el-table-column label="AI盒子" align="center">
              <template #default="scope">
                <el-button text :icon="Edit" @click="onOpenAiBoxDialog(scope.row)"> 修改 </el-button>
                <el-button text :icon="Message" @click="onOpen2Dialog('box', scope.row)">
                  AI盒子
                </el-button>
              </template>
            </el-table-column>

            <el-table-column label="监控球机" align="center">
              <template #default="scope">
                <el-button text :icon="Edit" @click="onRowContext(scope.row,$event)"> 修改 </el-button> 
                <el-button text :icon="Message" @click="onOpen2Dialog('ip_camera', scope.row)">
                  监控球机
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

    <!-- 编辑AI盒子 -->
    <edit-box
      :label-width="120"
      title="编辑AI盒子信息"
      ref="editBoxRef"
      @saved="getData"
      style="width: 60%; height: 60%"
    ></edit-box>

    <edit-camera
      :label-width="120"
      title="编辑监控球机信息"
      ref="editCameraRef"
      @saved="getData"
      style="width: 60%; height: 70%"
    ></edit-camera>
    <ec-context-menu ref="contextMenuRef" @checked="onCheckedMenu"></ec-context-menu>
    <edit-router
      :label-width="100"
      title="编辑路由器信息"
      ref="editRouterRef"
      @saved="getData"
      style="width: 60%; height: 70%"
    ></edit-router>

    <!-- 编辑监控球机 -->
    <edit-ip-camera
      ref="editIpCameraRef"
      :allow-modify-site="false"
      @saved="getData"
    ></edit-ip-camera>

    <!-- 详细信息 -->
    <diaglog-detail
      ref="diaglogRef"
      :column="3"
      :title="form2.title"
      :data="form2.data"
      style="width: 70%; height: 76%"
    ></diaglog-detail>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
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
  VideoCamera
} from '@element-plus/icons-vue'
import * as api from '../api/site' 
import * as  ipCamera_Api from '../api/site/ipCamera'
import * as dev_api from '../api/device'
import * as res_api from '../api'
import * as t from '../store/types/devices'
import { detailsInfo } from '../components/details'
import { imgVideo, types } from '../components/player'
import { str2Obj, createStateEndDatetime } from '../utils'
import { showLoading, closeLoading } from '../components/Logining'

 
import editBox from '../components/biz/src/editBox'
import editCamera from '../components/biz/src/editCamera'
import EcContextMenu, { type ContextMenuItem } from '../components/common/EcContextMenu'
import editRouter from '../components/biz/src/editRouter'
import diaglogDetail from '../components/common/diaglogDetail'

interface TableRow {
  id: number
  name: string
  postionInfo: string
  deviceCode: string
  deviceDesc: string
  createTime: string
  updateTime: string
}
interface Query extends IpageParam {
  name: string
  category?: number
  datetimes: Array<string>
}
interface table_module {
  query: Query
  moreOption: boolean
  data: TableRow[]
  currentRow?: TableRow
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

const queryDeviceDetailInfo = (siteId: number, deviceType: string) => {
  return api.getDetailInfo(siteId, deviceType)
}

//编辑AIBOX
const editBoxRef = ref<InstanceType<typeof editBox>>()
const onOpenAiBoxDialog = (row: TableRow) => {
  editBoxRef.value?.openDialog(row.id)
}
//编辑路由器信息
const editRouterRef = ref<InstanceType<typeof editRouter>>()
const onOpenRouterDialog = (row: TableRow) => {
  editRouterRef.value?.openDialog(row.id)
}
//编辑球机
const editCameraRef = ref<InstanceType<typeof editCamera>>()
const onOpenCameraDialog = (sietId:number,ipCameraId?:number) => {
  editCameraRef.value?.openDialog(sietId,ipCameraId)
}
//右键菜单
const contextMenuRef = ref<InstanceType<typeof EcContextMenu>>()
const onRowContext = (row:TableRow, event: any) => {
  event.preventDefault() //阻止鼠标右键默认行为
  ipCamera_Api.select_svc(row.id ).then((res) => {
    //有数据  
     let slectItem={id:-1,name:"新增"}
    if(res.data&&res.data.length>0) res.data.push(slectItem)
    else res.data=[slectItem] 
    contextMenuRef.value?.open(res.data,event,row.id )
  })
}
const onCheckedMenu = (index: number, item: ContextMenuItem,siteId:number) => { 
  if (item.id==-1)onOpenCameraDialog(siteId)
  else if (typeof item.id=="number") onOpenCameraDialog(siteId,item.id)
} 
//**详细下信息 */
const diaglogRef = ref<InstanceType<typeof diaglogDetail>>()

interface dataContent {
  name: string
  data: any
}
interface DataType {
  title: string
  data: dataContent[]
}

let form2 = ref<DataType>({
  title: '',
  data: []
})
const onOpen2Dialog = (category: string, row: TableRow) => {
  if (category == 'box') form2.value.title = '盒子信息'
  else if (category == 'router') form2.value.title = '路由器信息'
  else form2.value.title = '违停球信息'
  queryDeviceDetailInfo(row.id, category).then((res) => {
    if (res.code == 0) {
      let data = []
      for (let i = 0; i < res.data.length; i++)
        data.push({ name: res.data[i].name + '信息', data: res.data[i] })
      if (res.data.length == 0) ElMessage.warning('未找到关联设备')
      else diaglogRef.value?.openDiaLog(), (form2.value.data = data)
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
</style>
