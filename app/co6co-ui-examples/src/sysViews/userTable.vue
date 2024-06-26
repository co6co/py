<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input
            style="width: 160px"
            v-model="table_module.query.name"
            placeholder="用户名"
            class="handle-input mr10"
          ></el-input>
          <EnumSelect
            :data="selectData"
            v-model="table_module.query.state"
            style="width: 160px"
            placeholder="状态"
          ></EnumSelect>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
          <el-button
            type="primary"
            :icon="Plus"
            v-permiss="getPermissKey(ViewFeature.add)"
            @click="onOpenDialog(FormOperation.add)"
            >新增</el-button
          >
        </div>
      </el-header>
      <el-main>
        <el-scrollbar>
          <el-table
            :data="table_module.data"
            border
            class="table"
            ref="multipleTable"
            header-cell-class-name="table-header"
          >
            <el-table-column label="序号" width="55" align="center">
              <template #default="scope"> {{ scope.$index + 1 }}</template>
            </el-table-column>

            <el-table-column prop="userName" label="用户名" align="center"></el-table-column>
            <el-table-column label="所属用户组" prop="groupName"> </el-table-column>
            <el-table-column label="状态" align="center">
              <template #default="scope">
                <el-tag :type="getTagType(scope.row.state)">
                  {{ getName(scope.row.state) }}
                </el-tag>
              </template>
            </el-table-column>

            <el-table-column prop="createTime" label="注册时间"></el-table-column>
            <el-table-column label="操作" width="405" align="center">
              <template #default="scope">
                <el-button
                  text
                  :icon="Edit"
                  @click="onOpenDialog(FormOperation.edit, scope.row)"
                  v-permiss="getPermissKey(ViewFeature.edit)"
                >
                  编辑
                </el-button>
                <el-button
                  text
                  :icon="Setting"
                  @click="onOpenAssDiaglog(scope.row)"
                  v-permiss="getPermissKey(ViewFeature.associated)"
                >
                  关联
                </el-button>
                <el-button
                  text
                  :icon="Delete"
                  class="red"
                  @click="onDelete(scope.row.id, scope.row.userName)"
                  v-permiss="getPermissKey(ViewFeature.del)"
                >
                  删除
                </el-button>
                <el-button
                  text
                  :icon="Compass"
                  @click="onOpenResetDialog(scope.$index, scope.row)"
                  v-permiss="getPermissKey(ViewFeature.reset)"
                >
                  重置密码
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
            @current-change="handlePageChange"
          ></el-pagination>
        </div>
      </el-footer>
    </el-container>
    <!--编辑-->

    <modify-diaglog
      :title="table_module.diaglogTitle"
      ref="modifyDiaglogRef"
      @saved="getData"
    ></modify-diaglog>

    <!--重置密码-->
    <reset-pwd-diaglog ref="resetPwdDiaglogRef" title="重置密码"></reset-pwd-diaglog>
    <associated-diaglog
      ref="associatedDiaglogRef"
      style="width: 30%"
      title="关联角色"
    ></associated-diaglog>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox, ElTable, type FormRules, type FormInstance } from 'element-plus'
import { Delete, Edit, Search, Compass, Plus, Setting } from '@element-plus/icons-vue'
import * as icons from '@element-plus/icons-vue'
import { useState } from '../hook/sys/useUserSelect'
import api, { association_service as ass_api } from '../api/sys/user'
import modifyDiaglog, { type Item } from '../components/sys/modifyUser'
import resetPwdDiaglog from '../components/sys/resetPwd'

import {
  EnumSelect,
  showLoading,
  closeLoading,
  type IPageParam,
  type Table_Module_Base,
  FormOperation
} from 'co6co'
import useDelete from '../hook/sys/useDelete'

import associatedDiaglog from '../components/sys/associated'
import { usePermission, ViewFeature } from '../hook/sys/useRoute'

interface IQueryItem extends IPageParam {
  name?: string
  state?: number
}
interface Table_Module extends Table_Module_Base {
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
const { selectData, getName, getTagType } = useState()
const { getPermissKey } = usePermission()
// 获取表格数据
const getData = () => {
  showLoading()
  api
    .get_table_svc(table_module.query)
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
  table_module.query.pageIndex = 1
  getData()
}
// 分页导航
const handlePageChange = (val: number) => {
  table_module.query.pageIndex = val
  getData()
}
//增加/修改
const modifyDiaglogRef = ref<InstanceType<typeof modifyDiaglog>>()
const onOpenDialog = (operation: FormOperation, row?: Item) => {
  table_module.diaglogTitle =
    operation == FormOperation.add ? '增加用户' : `编辑[${row?.userName}]用户`
  table_module.currentItem = row
  modifyDiaglogRef.value?.openDialog(operation, row)
}
//重置密码
const resetPwdDiaglogRef = ref<InstanceType<typeof resetPwdDiaglog>>()
const onOpenResetDialog = (_: number, row?: Item) => {
  table_module.currentItem = row
  resetPwdDiaglogRef.value?.openDialog(row)
}
//删除
const { onDelete } = useDelete(api.del_svc, getData)
onMounted(() => {
  getData()
})

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
</script>

<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
