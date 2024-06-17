<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-select
            style="width: 160px"
            clearable
            class="mr10"
            v-model="table_module.query.userId"
            placeholder="用户名"
          >
            <el-option
              v-for="(item, index) in userSelect"
              :key="index"
              :label="item.name"
              :value="item.id"
            />
          </el-select>

          <el-select
            style="width: 160px"
            clearable
            class="mr10"
            v-model="table_module.query.status"
            placeholder="状态"
          >
            <el-option
              v-for="(item, index) in stateList"
              :key="item.key"
              :label="item.label"
              :value="item.value"
            />
          </el-select>

          <el-button type="primary" :icon="Search" @click="onSearch">查询</el-button>
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
            <el-table-column label="序号" width="130" align="center" :show-overflow-tooltip="true">
              <template #default="scope">
                <span>{{ getTableIndex(table_module.query, scope.$index) }}</span>
              </template>
            </el-table-column>
            <el-table-column
              prop="jobId"
              label="任务"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="userName"
              label="用户名"
              width="119"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column label="状态" prop="status" sortable="custom" align="center">
              <template #default="scope">
                <el-tag :type="statue2TagType(scope.row.status)">
                  {{ getStateName(scope.row.status)?.label }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="recordNum" sortable="custom" label="接单">
              <template #default="scope">
                {{ getNumberDesc(scope.row.recordNum) }}
              </template>
            </el-table-column>
            <el-table-column prop="passNum" sortable="custom" label="确警">
              <template #default="scope">
                {{ getNumberDesc(scope.row.passNum) }}
              </template>
            </el-table-column>
            <el-table-column prop="passIssueNum" sortable="custom" label="通过下发">
              <template #default="scope">
                {{ getNumberDesc(scope.row.passIssueNum) }}
              </template>
            </el-table-column>
            <el-table-column prop="unpassNum" sortable="custom" label="误警">
              <template #default="scope">
                {{ getNumberDesc(scope.row.unpassNum) }}
              </template>
            </el-table-column>
            <el-table-column prop="ignoreNum" sortable="custom" label="忽略">
              <template #default="scope">
                {{ getNumberDesc(scope.row.ignoreNum) }}
              </template>
            </el-table-column>
            <el-table-column prop="completTate" sortable="custom" label="完成率">
              <template #default="scope">
                {{ getNumberDesc(scope.row.completTate, '%') }}
              </template>
            </el-table-column>
            <el-table-column prop="spendTime" sortable="custom" label="花费时间">
              <template #default="scope">
                {{ getNumberDesc(scope.row.spendTime, '秒') }}
              </template>
            </el-table-column>

            <el-table-column
              prop="createTime"
              sortable="custom"
              label="创建时间"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="updateTime"
              sortable="custom"
              label="更新时间"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column label="操作" align="center" width="160px">
              <template #default="scope">
                <!--<router-link :to="{name:'userJobsDetail',query:{'id':scope.row.id}}">查看详情</router-link>-->
                <router-link
                  v-permiss="getPermissKey(ViewFeature.view)"
                  :to="{ path: '/userJobsDetail/' + scope.row.id }"
                  >查看详情</router-link
                >
                <el-button
                  text
                  :icon="Delete"
                  class="red"
                  @click="onDelete(scope.$index, scope.row)"
                  v-permiss="getPermissKey(ViewFeature.del)"
                >
                  删除
                </el-button>
              </template>
            </el-table-column>
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
  Compass,
  Plus,
  Download,
  ArrowUp,
  ArrowDown
} from '@element-plus/icons-vue'
import { showLoading, closeLoading } from '../components/Logining'
import { createStateEndDatetime } from 'co6co'
import { getTableIndex } from 'co6co'
import * as api from '../api/boat/jobs'
import useUserSelect from '../hook/useUserSelect'
import useJobState from '../hook/useJobState'
import type * as api_types from 'co6co'
//import router from '@/router'
import { useRouter } from 'vue-router'
import { usePermission, ViewFeature } from '../hook/sys/useRoute'
const { getPermissKey } = usePermission()
interface TableRow {
  id: number
  userId: number
  jobId: number
  userName: string
}

interface Query extends api_types.IPageParam {
  datetimes: Array<string>
  userId?: number
  status?: number
}
interface Table_Module extends api_types.Table_Module_Base {
  query: Query
  moreOption: boolean
  data: TableRow[]
  currentRow?: TableRow
}
const table_module = reactive<Table_Module>({
  query: {
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
const getNumberDesc = (value?: number, unit: string = '条') => {
  if (value == undefined) return '0' + unit
  else return value + unit
}
const { userSelect } = useUserSelect()
const { stateList, getStateName, statue2TagType } = useJobState()

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
// 分页导航
const onPageChange = (pageIndex: number) => {
  let totalPage = Math.ceil(table_module.pageTotal / table_module.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error('已经是第一页了')
  else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了')
  else (table_module.query.pageIndex = pageIndex), getData()
}

const onTableSelect = (row: any) => {}
const router = useRouter()
const onGoto = (index: number, row: TableRow) => {
  // params 为 路由反模式,不建议这样使用，
  router.push({ name: 'userJobsDetail', params: { list: JSON.stringify(row) } })
}
const onDelete = (index: number, row: TableRow) => {
  // 二次确认删除
  ElMessageBox.confirm(`确定要删除"${row.userName}-${row.jobId}"吗？`, '提示', {
    type: 'warning'
  })
    .then(() => {
      showLoading()
      api
        .del_svc(row.id)
        .then((res) => {
          if (res.code == 0) ElMessage.success('删除成功'), getData()
          else ElMessage.error(`删除失败:${res.message}`)
        })
        .finally(() => {
          closeLoading()
        })
    })
    .catch(() => {})
}
</script>

<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
