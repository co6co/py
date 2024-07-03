<!--正在抽象中-->
<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <slot name="header" />
          <!--header 内容-->
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
            <slot name="tableContent" />
          </el-table>
        </el-scrollbar>
        <slot />
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
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, watchEffect, computed } from 'vue'
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
  Compass,
  Plus,
  Download,
  ArrowUp,
  ArrowDown
} from '@element-plus/icons-vue'

import {
  createStateEndDatetime,
  showLoading,
  closeLoading,
  type IPageParam,
  type Table_Module_Base
} from 'co6co'

interface TableRow {
  id: number
  uuid: string
  alarmType: string
  videoUid: string
  rawImageUid: string
  markedImageUid: string
  alarmTime: string
  createTime: string
}
interface AlertCategory {
  alarmType: string
  desc: string
}
interface Query extends IPageParam {
  datetimes: Array<string>
  alarmType: String
}
interface Table_Module extends Table_Module_Base {
  query: Query
  moreOption: boolean
  data: TableRow[]
  currentRow?: TableRow
  categoryList: AlertCategory[]
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
  pageTotal: -1,
  categoryList: []
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
const onRefesh = () => {
  getData()
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
}

// 分页导航
const onPageChange = (pageIndex: number) => {
  let totalPage = Math.ceil(table_module.pageTotal / table_module.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error('已经是第一页了')
  else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了')
  else (table_module.query.pageIndex = pageIndex), getData()
}

const onTableSelect = (row: any) => {}
</script>

<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
