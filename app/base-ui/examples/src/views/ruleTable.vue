<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box" style="text-align: right">
          <el-button type="primary" :icon="Search" @click="onSearch">刷新</el-button>
          <el-button v-permiss="getPermissKey(ViewFeature.add)" type="primary" :icon="Plus" @click="onOpenDialog()">新增</el-button>
          <el-button v-permiss="getPermissKey(ViewFeature.effective)"
            type="warning"
            :disabled="!table_module.priorityChanged" 
            :icon="Connection"
            @click="onPriorityChanged"
            >优先级立即生效</el-button
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
              prop="name"
              label="规则名"
              width="119"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column label="是否启用" prop="enable" width="110" sortable="custom">
              <template #default="scope">
                <el-tag :type="getEleTagTypeByBoolean(scope.row.enable)"
                  >{{ scope.row.enable == 0 ? '禁用' : '启用' }}
                </el-tag></template
              >
            </el-table-column>

            <el-table-column
              prop="code"
              label="编码"
              width="119"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="confidence"
              label="置信度"
              width="120"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="priority"
              label="优先级" 
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>

            <el-table-column
              prop="baseRule"
              label="基础规则" 
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="aiAuditAccuracy"
              label="AI审核正确率"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="aiNetName"
              label="AI模型名称"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column
              prop="aiStatisticsNum"
              label="AI统计"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            
            <el-table-column label="人工复审" prop="manualReview" width="110" sortable="custom">
              <template #default="scope">
                <el-tag :type="getEleTagTypeByBoolean(scope.row.manualReview)"
                  >{{ scope.row.manualReview +(scope.row.manualReview == 1 ? '-复审' : '-AI审') }}
                </el-tag></template
              >
            </el-table-column>
            <el-table-column label="操作" width="194" align="center">
              <template #default="scope">
                <el-button text :icon="Edit" @click="onOpenDialog(scope.row)" v-permiss="getPermissKey(ViewFeature.edit)">
                  编辑
                </el-button>
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
    <edit-rule ref="editViewRef" @saved="onAddEditSaved"></edit-rule>
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
  Connection,
  ArrowUp,
  ArrowDown
} from '@element-plus/icons-vue'
import { showLoading, closeLoading } from '../components/Logining'
import { createStateEndDatetime } from '../utils'
import { getTableIndex } from '../utils/tables'
import * as api from '../api/boat/rules'
import * as dp_api from '../api/pd'
import * as api_types from '../api/types'
import EditRule ,{type Item,type FromData} from '../components/biz/EditRule'
import {getEleTagTypeByBoolean,type IPageParam} from '../api/types' 
import useNotifyAudit,{NotifyType} from '../hook/useNotifyAudit'
import {usePermission,ViewFeature} from '../hook/sys/useRoute'
const {getPermissKey}= usePermission()
interface Query extends IPageParam {
  datetimes: Array<string>
  alarmType: String
}
interface Table_Module  extends api_types.Table_Module_Base{
  query: Query
  moreOption: boolean
  data: Item[]
  currentRow?: Item 
  priorityTemp?:number;
  priorityChanged:boolean
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
  priorityChanged:false
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
        table_module.pageTotal=res.total || 0

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
const onTableSelect = (row: Item) => {}
const editViewRef = ref<typeof EditRule>()
const onOpenDialog = (row?: Item) => {
  // 临时保存优先级
  table_module.priorityTemp=row?.priority
  //有记录编辑无数据增加
  editViewRef.value?.onOpenDialog(row ? api_types.Operation.Edit : api_types.Operation.Add, row)
}
const onAddEditSaved=(saveData:FromData)=>{ 
  if (table_module.priorityTemp!=saveData.priority && !table_module.priorityChanged){
    table_module.priorityChanged=true
  }
  getData()
}
 
const { notifyAuditSystem}=useNotifyAudit()
const onPriorityChanged=()=>{
  notifyAuditSystem({type: NotifyType.rule_priority_changed,state: true, failMessage:"规则优先级通知失败",message:"规则优先级通知成功"})
}
 
const onDelete = (index: number, row: Item) => {
  // 二次确认删除
  ElMessageBox.confirm(`确定要删除"${row.name}"吗？`, '提示', {
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
