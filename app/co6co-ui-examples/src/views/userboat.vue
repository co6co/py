<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box" style="text-align: right">
          <!-- <el-button
            type="primary"
            style="float: right; margin-right: 15px"
            :icon="Plus"
            @click="onOpenDialog()"
            >新增</el-button
          >
          -->
          <el-button type="primary" :icon="Search" @click="onSearch">刷新</el-button>
          <el-button
            v-permiss="getPermissKey(routeHook.ViewFeature.associated)"
            type="primary"
            :icon="Setting"
            @click="onOpenDialog2()"
            >关联</el-button
          >
          <el-button
            v-permiss="getPermissKey(routeHook.ViewFeature.check)"
            type="primary"
            :icon="View"
            @click="onOpenDialog3()"
            >检查未关联</el-button
          >
        </div>
      </el-header>
      <el-main>
        <el-scrollbar>
          <!--table 内容-->
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
            <el-table-column label="序号" width="119" align="center" :show-overflow-tooltip="true">
              <template #default="scope">
                <span>{{ getTableIndex(table_module.query, scope.$index) }}</span>
              </template>
            </el-table-column>
            <el-table-column
              prop="userName"
              label="用户名"
              width="119"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column
              prop="boatName"
              label="船名称"
              width="119"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column
              prop="boatSerial"
              label="船序列号"
              width="200"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column prop="createTime" sortable="custom" label="创建时间"></el-table-column>
          </el-table>
        </el-scrollbar>
      </el-main>
      <el-footer>
        <div class="pagination">
          <!--分页内容-->
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
        </div>
      </el-footer>
    </el-container>
    <!--<edit-user-boat ref="editViewRef" @saved="getData()"></edit-user-boat>-->
    <edit-user-boats ref="editViewsRef" @saved="onSaved"></edit-user-boats>
    <user-boat-check ref="userBoatCheckRef"></user-boat-check>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import {
  ElMessage,
  ElMessageBox,
  type FormRules,
  type FormInstance,
  ElLoading,
  ElTable
} from 'element-plus'
import {
  Delete,
  Edit,
  Search,
  View,
  Plus,
  Setting,
  ArrowUp,
  ArrowDown
} from '@element-plus/icons-vue'

import {
  createStateEndDatetime,
  getTableIndex,
  showLoading,
  closeLoading,
  Operation,
  type IPageParam,
  type Table_Module_Base
} from 'co6co'

import * as api from '../api/boat'

import EditUserBoat, { type Item } from '../components/biz/EditUserBoat'
import editUserBoats from '../components/biz/editUserBoats'
import userBoatCheck from '../components/biz/UserBoatCheck'

import { routeHook } from 'co6co-right'
const { getPermissKey } = routeHook.usePermission()

interface Query extends IPageParam {
  datetimes: Array<string>
  alarmType: String
}
interface Table_Module extends Table_Module_Base {
  query: Query
  moreOption: boolean
  data: Item[]
  currentRow?: Item
}
const table_module = reactive<Table_Module>({
  query: {
    alarmType: '',
    datetimes: [],
    pageIndex: 1,
    pageSize: 15,
    order: 'asc',
    orderBy: ''
  },
  moreOption: false,
  data: [],
  pageTotal: -1
})
const tableInstance = ref<InstanceType<typeof ElTable>>()
// 排序
const onColChange = (column: any) => {
  table_module.query.order = column.order === 'descending' ? 'desc' : 'asc'
  table_module.query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}

const moreIcon = computed(() => {
  if (table_module.moreOption) return ArrowUp
  else return ArrowDown
})
const setDatetime = (t: number, i: number) => {
  table_module.query.datetimes = createStateEndDatetime(t, i)
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
    .get_list_svc(table_module.query)
    .then((res) => {
      if (res.code == 0) {
        table_module.data = res.data
        table_module.pageTotal = res.total || 0
      } else {
        ElMessage.error(res.message)
      }
    })
    .finally(() => closeLoading())
}

getData()
const onSaved = (userId: number) => {
  getData()
}
// 分页导航
const onPageChange = (pageIndex: number) => {
  let totalPage = Math.ceil(table_module.pageTotal / table_module.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error('已经是第一页了')
  else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了')
  else (table_module.query.pageIndex = pageIndex), getData()
}

const onTableSelect = (row: any) => {}
//单个编辑
const editViewRef = ref<InstanceType<typeof EditUserBoat>>()
const onOpenDialog = (row?: any) => {
  //有记录编辑无数据增加
  editViewRef.value?.onOpenDialog(row ? Operation.Edit : Operation.Add, row)
}
//多个编辑2
const editViewsRef = ref<InstanceType<typeof editUserBoats>>()
const onOpenDialog2 = () => {
  editViewsRef.value?.onOpenDialog()
}
//检查是否关联
const userBoatCheckRef = ref<InstanceType<typeof userBoatCheck>>()
const onOpenDialog3 = () => {
  userBoatCheckRef.value?.onOpenDialog()
}
</script>

<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
