<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input v-model="query.name" placeholder="名称" class="handle-input mr10"></el-input>
          <el-input
            v-model="query.boatPosNumber"
            placeholder="部位编号"
            class="handle-input mr10"
          ></el-input>
          <el-select style="width: 160px" class="mr10" v-model="query.type" placeholder="请选择">
            <el-option
              v-for="item in state_store.group"
              :key="item.key"
              :label="item.label"
              :value="item.key"
            />
          </el-select>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
        </div>
      </el-header>
      <el-main>
        <el-scrollbar>
          <el-table
            :data="tableData"
            @sort-change="onColChange"
            border
            class="table"
            ref="multipleTable"
            header-cell-class-name="table-header"
          >
            <el-table-column label="序号" width="55" align="center">
              <template #default="scope"> {{ scope.$index + 1 }} </template>
            </el-table-column>
            <el-table-column
              prop="name"
              label="名称"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column label="分组类型" sortable="custom" :show-overflow-tooltip="true">
              <template #default="scope">
                {{ scope.row.groupType }} {{ state_store.getGroupItem(scope.row.groupType)?.label }}
              </template>
            </el-table-column>
            <el-table-column label="序列号" sortable="custom" :show-overflow-tooltip="true">
              <template #default="scope">
                {{ scope.row.ipCameraSerial || scope.row.boatSerial }}
              </template>
            </el-table-column>
            <el-table-column
              label="部位编号"
              sortable="custom"
              :show-overflow-tooltip="true"
              prop="boatPosNumber"
            ></el-table-column>
            <el-table-column
              prop="updateTime"
              label="更新时间"
              sortable="custom"
              :show-overflow-tooltip="true"
            ></el-table-column>
            <el-table-column label="操作" width="316" align="center">
              <template #default="scope">
                <el-button
                  v-permiss="getPermissKey(ViewFeature.settingNo)"
                  text
                  :icon="Setting"
                  v-if="state_store.allowSetting(scope.row.groupType)"
                  @click="onOpenEditDialog(scope.$index, scope.row)"
                >
                  设置编号
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
            layout="total, prev, pager, next"
            :current-page="query.pageIndex"
            :page-size="query.pageSize"
            :total="pageTotal"
            @current-change="onCurrentPageChange"
          ></el-pagination>
        </div>
      </el-footer>
    </el-container>
    <edit-group ref="editViewRef" @saved="getData()"></edit-group>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue'
import { ElMessage, ElMessageBox, type FormRules, type FormInstance } from 'element-plus'
import { Delete, Edit, Search, Compass, Plus, Setting } from '@element-plus/icons-vue'
import { queryList_svc } from '../api/group'
import { group_state_store } from '../store/group'
import editGroup, { type Item } from '../components/biz/editGroup'
import * as api_types from 'co6co'
import { showLoading, closeLoading } from '../components/Logining'
import { usePermission, ViewFeature } from '../hook/sys/useRoute'
const { getPermissKey } = usePermission()
const state_store = group_state_store()
state_store.refesh().then((res) => {})
interface QueryType {
  name?: string
  type?: string
  boatPosNumber?: string
  pageIndex: number
  pageSize: number
  order: 'asc' | 'desc'
  orderBy: string
}
const query = reactive<QueryType>({
  pageIndex: 1,
  pageSize: 10,
  order: 'asc',
  orderBy: ''
})
const tableData = ref<any[]>([])
const pageTotal = ref(0)
// 获取表格数据
const getData = () => {
  showLoading()
  queryList_svc(query)
    .then((res) => {
      if (res.code == 0) {
        tableData.value = res.data
        pageTotal.value = res.total || -1
      } else {
        ElMessage.error(res.message)
      }
    })
    .finally(() => {
      closeLoading()
    })
}
getData()
const onColChange = (column: any) => {
  //console.info(column)
  query.order = column.order === 'descending' ? 'desc' : 'asc'
  query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}
// 查询操作
const onSearch = () => {
  query.pageIndex = 1
  getData()
}
// 分页导航
const onCurrentPageChange = (val: number) => {
  query.pageIndex = val
  getData()
}
const editViewRef = ref<typeof editGroup>()
const onOpenEditDialog = (index: number, row: Item) => {
  editViewRef.value?.onOpenDialog(row ? api_types.Operation.Edit : api_types.Operation.Add, row)
}
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
