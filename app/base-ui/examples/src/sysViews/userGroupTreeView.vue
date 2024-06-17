<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input
            style="width: 160px"
            clearable
            v-model="table_module.query.name"
            placeholder="组名"
            class="handle-input mr10"
          ></el-input>
          <el-input
            style="width: 160px"
            clearable
            v-model="table_module.query.code"
            placeholder="编码"
            class="handle-input mr10"
          ></el-input>

          <el-button type="primary" :icon="Search" @click="onSearch">查询</el-button>
          <el-button
            v-permiss="getPermissKey(ViewFeature.add)"
            type="primary"
            :icon="Plus"
            @click="onOpenDialog(api_types.FormOperation.add)"
            >新增</el-button
          >
        </div>
      </el-header>
      <el-container>
        <el-container>
          <el-main>
            <el-scrollbar>
              <el-table
                :data="table_module.data"
                @sort-change="onColChange"
                border
                class="table"
                header-cell-class-name="table-header"
                row-key="id"
                :tree-props="{ children: 'children' }"
              >
                <el-table-column label="序号" width="100" align="center">
                  <template #default="scope"> {{ scope.$index + 1 }} </template>
                </el-table-column>
                <el-table-column
                  prop="name"
                  label="名称"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                ></el-table-column>
                <el-table-column
                  width="100"
                  label="父节点"
                  prop="parentId"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                >
                  <template #default="scope"> {{ getName(scope.row.parentId) || '-' }} </template>
                </el-table-column>
                <el-table-column
                  width="80"
                  prop="code"
                  label="代码"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                ></el-table-column>

                <el-table-column
                  width="80"
                  prop="order"
                  label="排序"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                ></el-table-column>

                <el-table-column
                  width="156"
                  prop="createTime"
                  label="创建时间"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                ></el-table-column>
                <el-table-column
                  width="156"
                  prop="updateTime"
                  label="更新时间"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                ></el-table-column>
                <el-table-column label="操作" width="315" align="center">
                  <template #default="scope">
                    <el-button
                      v-permiss="getPermissKey(ViewFeature.edit)"
                      text
                      :icon="Setting"
                      @click="onOpenDialog(api_types.FormOperation.edit, scope.row)"
                    >
                      编辑
                    </el-button>
                    <el-button
                      v-permiss="getPermissKey(ViewFeature.associated)"
                      text
                      :icon="Setting"
                      @click="onOpenAssDiaglog(scope.row)"
                    >
                      关联角色
                    </el-button>
                    <el-button
                      v-permiss="getPermissKey(ViewFeature.del)"
                      text
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
                :current-page="table_module.query.pageIndex"
                :page-size="table_module.query.pageSize"
                :total="table_module.pageTotal"
                @current-change="onPageChange"
              ></el-pagination>
            </div>
          </el-footer>
        </el-container>
      </el-container>
    </el-container>

    <modify-diaglog
      :title="table_module.diaglogTitle"
      ref="modifyDiaglogRef"
      @saved="onLoadData"
    ></modify-diaglog>

    <associated-diaglog
      style="width: 30%"
      title="关联角色"
      ref="associatedDiaglogRef"
    ></associated-diaglog>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox, type FormRules, type FormInstance } from 'element-plus'
import {
  Delete,
  Sunny,
  Edit,
  Search,
  Compass,
  Plus,
  Setting,
  Connection
} from '@element-plus/icons-vue'
import * as api_types from 'co6co'

import modifyDiaglog, { type Item } from '../components/sys/modifyUserGroup'
import { showLoading, closeLoading } from '../components/Logining'
import useUserGroupSelect from '../hook/sys/useUserGroupSelect'
import associatedDiaglog from '../components/sys/associated'
import api, { association_service as ass_api } from '../api/sys/userGroup'
import { usePermission, ViewFeature } from '../hook/sys/useRoute'
const { getPermissKey } = usePermission()
interface IQueryItem extends api_types.IPageParam {
  name?: string
  code?: string
  parentId?: number
}

interface Table_Module extends api_types.Table_Module_Base {
  query: IQueryItem
  data: Item[]
  currentItem?: Item
}
const table_module = reactive<Table_Module>({
  query: {
    pageIndex: 1,
    pageSize: 10,
    order: 'asc',
    orderBy: ''
  },
  data: [],
  pageTotal: -1,
  diaglogTitle: ''
})
const { selectData, refresh, getName } = useUserGroupSelect()
// 获取表格数据
const getData = () => {
  showLoading()
  api
    .get_tree_table_svc(table_module.query)
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

// 查询操作
const onSearch = () => {
  getData()
}

const onColChange = (column: any) => {
  table_module.query.order = column.order === 'descending' ? 'desc' : 'asc'
  table_module.query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}
const onPageChange = (val: number) => {
  table_module.query.pageIndex = val
  getData()
}
const modifyDiaglogRef = ref<InstanceType<typeof modifyDiaglog>>()
const onOpenDialog = (operation: api_types.FormOperation, row?: Item) => {
  table_module.diaglogTitle = operation == api_types.FormOperation.add ? '增加用户组' : '编辑用户组'
  table_module.currentItem = row
  modifyDiaglogRef.value?.openDialog(operation, row)
}
//关联
const associatedDiaglogRef = ref<InstanceType<typeof associatedDiaglog>>()
const onOpenAssDiaglog = (row?: Item) => {
  table_module.currentItem = row
  associatedDiaglogRef.value?.openDialog(
    row!.id,
    ass_api.get_association_svc,
    ass_api.save_association_svc
  )
}

const onLoadData = () => {
  refresh()
  getData()
  modifyDiaglogRef.value?.update()
}
// 删除操作
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
          if (res.code == 0) ElMessage.success('删除成功'), onLoadData()
          else ElMessage.error(`删除失败:${res.message}`)
        })
        .finally(() => {
          closeLoading()
        })
    })
    .catch(() => {})
}
onMounted(() => {
  onLoadData()
})
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
