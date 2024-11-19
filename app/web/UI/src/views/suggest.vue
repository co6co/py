<template>
  <div class="container-layout c-container">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input
            style="width: 160px"
            clearable
            v-model="table_module.query.title"
            placeholder="名称"
            class="handle-input"
          ></el-input>
          <ElSelect style="width: 160px" v-model="table_module.query.category" placeholder="类别">
            <ElOption
              v-for="(item, index) in dictHook.selectData.value"
              :key="index"
              :label="item.name"
              :value="Number(item.value)"
            ></ElOption>
          </ElSelect>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
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
            <el-table-column
              prop="title"
              label="名称"
              sortable="custom"
              align="center"
              width="180"
              :show-overflow-tooltip="true"
            />

            <el-table-column label="类别" sortable="custom" prop="category" align="center">
              <template #default="scope">
                <el-tag>
                  {{ dictHook.getName(scope.row.category) }}
                </el-tag>
              </template>
            </el-table-column>

            <el-table-column label="状态" sortable="custom" prop="state" align="center">
              <template #default="scope">
                {{ suggestStateHook.getName(scope.row.state) }}
              </template>
            </el-table-column>
            <el-table-column
              prop="createTime"
              label="创建时间"
              sortable="custom"
              :show-overflow-tooltip="true"
            />

            <el-table-column label="操作" width="320" align="center" fixed="right">
              <template #default="scope">
                <el-button
                  text
                  :icon="View"
                  @click="onOpenDialog(scope.row)"
                  v-permiss="getPermissKey(routeHook.ViewFeature.view)"
                >
                  详情
                </el-button>
                <el-button
                  text
                  :icon="View"
                  @click="onOpenReplyDialog(scope.row)"
                  v-permiss="getPermissKey(routeHook.ViewFeature.view)"
                >
                  回复
                </el-button>
                <el-button
                  text
                  :icon="Delete"
                  class="red"
                  @click="onDelete(scope.row.id, scope.row)"
                  v-permiss="getPermissKey(routeHook.ViewFeature.del)"
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
            @current-change="handlePageChange"
          ></el-pagination>
        </div>
      </el-footer>
    </el-container>

    <!--编辑-->
    <Diaglog
      :title="table_module.diaglogTitle"
      ref="detailDiaglogRef"
      @data-change="getData"
    ></Diaglog>
    <replyDiaglog
      :title="table_module.diaglogTitle"
      ref="replyDiaglogRef"
      @saved="getData"
    ></replyDiaglog>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive, onMounted } from 'vue'
import {
  ElTag,
  ElButton,
  ElInput,
  ElTable,
  ElTableColumn,
  ElContainer,
  ElMain,
  ElHeader,
  ElScrollbar,
  ElPagination,
  ElFooter
} from 'element-plus'
import { Delete, Edit, Search, Plus, View } from '@element-plus/icons-vue'

import { showLoading, closeLoading, type IPageParam, type Table_Module_Base } from 'co6co'
import { routeHook, deleteHook, useDictHook } from 'co6co-right'
import Diaglog from '../components/biz/detailSuggest'
import replyDiaglog from '../components/biz/reply'
import { get_table_svc, del_svc, Item } from '../api/biz/sugest'
import { DictTypeCodes } from '../api/app'

interface IQueryItem extends IPageParam {
  title?: string
  category?: number
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
const { getPermissKey } = routeHook.usePermission()
// 获取表格数据
const getData = () => {
  showLoading()
  get_table_svc(table_module.query)
    .then((res) => {
      table_module.data = res.data
      table_module.pageTotal = res.total || -1
    })
    .finally(() => {
      closeLoading()
    })
}

const dictHook = useDictHook.useDictSelect()
const suggestStateHook = useDictHook.useDictSelect()

onMounted(async () => {
  await dictHook.queryByCode(DictTypeCodes.SuggestType)
  await suggestStateHook.queryByCode(DictTypeCodes.SuggestState)
  getData()
})
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
const detailDiaglogRef = ref<InstanceType<typeof Diaglog>>()
const onOpenDialog = (row?: Item) => {
  table_module.diaglogTitle = `[${row?.title}]详情`
  table_module.currentItem = row
  if (row?.id) detailDiaglogRef.value?.openDialog(row.id)
}
//回复
const replyDiaglogRef = ref<InstanceType<typeof replyDiaglog>>()
const onOpenReplyDialog = (row?: Item) => {
  table_module.currentItem = row
  if (row?.id) replyDiaglogRef.value?.openDialog(row.id)
}
// 删除操作
const { deleteSvc } = deleteHook.default(del_svc, getData)
const onDelete = (_: number, row: Item) => {
  deleteSvc(row.id, row.title, true)
}
</script>
