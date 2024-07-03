<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input
            v-model="query.name"
            placeholder="任务名称"
            class="handle-input mr10"
          ></el-input>
          <el-select style="width: 160px" class="mr10" v-model="query.type" placeholder="请选择">
            <el-option
              v-for="item in from_attach_data.task_types"
              :key="item.key"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-select style="width: 160px" class="mr10" v-model="query.statue" placeholder="请选择">
            <el-option
              v-for="item in from_attach_data.tast_status"
              :key="item.key"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
        </div>
      </el-header>
      <el-main>
        <el-scrollbar>
          <el-table
            :data="tableData"
            border
            class="table"
            ref="multipleTable"
            header-cell-class-name="table-header"
          >
            <el-table-column label="序号" width="55" align="center">
              <template #default="scope"> {{ scope.$index }} </template>
            </el-table-column>
            <el-table-column prop="id" label="ID" width="90" align="center"></el-table-column>
            <el-table-column prop="name" label="名称"></el-table-column>
            <el-table-column label="类型">
              <template #default="scope">
                {{ from_attach_data.getTypeItem(scope.row.type)?.label }}
              </template>
            </el-table-column>
            <el-table-column label="状态" align="center">
              <template #default="scope">
                <el-tag :type="from_attach_data.statue2TagType(scope.row.status)">
                  {{ from_attach_data.getStateItem(scope.row.status)?.label }}
                </el-tag>
              </template>
            </el-table-column>

            <el-table-column prop="createTime" label="创建时间"></el-table-column>
            <el-table-column label="操作" width="316" align="center">
              <template #default="scope">
                <v-download
                  v-permiss="getPermissKey(routeHook.ViewFeature.download)"
                  :url="getDownloadUrl(scope.$index, scope.row)"
                  :file-name="`${scope.row.name}.zip`"
                ></v-download>
                <!--v-permiss="16"-->
                <el-button
                  text
                  v-permiss="getPermissKey(routeHook.ViewFeature.del)"
                  :icon="Delete"
                  class="red"
                  @click="onDelete(scope.$index, scope.row)"
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
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue'
import { ElMessage, ElMessageBox, type FormRules, type FormInstance } from 'element-plus'
import { Delete, Edit, Search, Compass, Plus, Download } from '@element-plus/icons-vue'
import { queryList_svc, del_svc } from '../api/tasks'

import { attach_data, type TaskAtachData } from '../store/tasks'

import { download as vDownload } from '../components/download'
import { showLoading, closeLoading } from 'co6co'
import { routeHook, download_svc } from 'co6co-right'
const { getPermissKey } = routeHook.usePermission()
let from_attach_data = reactive<TaskAtachData>(attach_data)
interface TableItem {
  id: number
  userName: string
  state: number
  roleId: number
}
interface QueryType {
  name?: string
  type?: number
  statue?: number
  pageIndex: number
  pageSize: number
}
const query = reactive<QueryType>({
  name: '',
  statue: undefined,
  type: undefined,
  pageIndex: 1,
  pageSize: 10
})
const tableData = ref<TableItem[]>([])
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

// 删除操作
const onDelete = (index: number, row: any) => {
  // 二次确认删除
  ElMessageBox.confirm(`确定要删除"${row.name}"任务吗？`, '提示', {
    type: 'warning'
  })
    .then(() => {
      del_svc(row.id)
        .then((res) => {
          if (res.code == 0) ElMessage.success('删除成功'), getData()
          else ElMessage.error(`删除失败:${res.message}`)
        })
        .finally(() => {})
    })
    .catch(() => {})
}

const getDownloadUrl = (index: number, row: any) => {
  return import.meta.env.VITE_TASK_PATH + row.data
}
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
