<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input
            style="width: 160px"
            clearable
            v-model="table_module.query.name"
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

          <EnumSelect
            :data="selectData"
            clearable
            v-model="table_module.query.state"
            style="width: 160px"
            placeholder="状态"
          ></EnumSelect>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
          <el-button
            type="primary"
            :icon="Plus"
            v-permiss="getPermissKey(routeHook.ViewFeature.add)"
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
            <el-table-column
              prop="name"
              label="名称"
              align="center"
              width="180"
              sortable="custom"
              :show-overflow-tooltip="true"
            />
            <el-table-column label="类别" sortable="custom" align="center">
              <template #default="scope">
                <el-tag>
                  {{ dictHook.getName(scope.row.category) }}
                </el-tag>
              </template>
            </el-table-column>

            <el-table-column label="公众号" sortable="custom" width="110" prop="openId">
              <template #default="scope">
                <el-tag>{{ store.getItem(scope.row.openId)?.name }} </el-tag></template
              >
            </el-table-column>
            <el-table-column label="状态" align="center" sortable="custom">
              <template #default="scope">
                <el-tag :type="getTagType(scope.row.state)">
                  {{ getName(scope.row.state) }}
                </el-tag>
              </template>
            </el-table-column>

            <el-table-column
              label="地址"
              align="left"
              sortable="custom"
              prop="url"
              :show-overflow-tooltip="true"
            >
              <template #default="scope">
                <el-icon :color="scope.row.color">
                  <component :is="scope.row.icon"></component>
                </el-icon>
                {{ scope.row.url }}
              </template>
            </el-table-column>

            <el-table-column
              prop="order"
              label="排序"
              align="center"
              width="90"
              :show-overflow-tooltip="true"
            />

            <el-table-column
              prop="createTime"
              label="创建时间"
              sortable="custom"
              :show-overflow-tooltip="true"
            />
            <el-table-column
              prop="updateTime"
              label="更新时间"
              sortable="custom"
              :show-overflow-tooltip="true"
            />
            <el-table-column label="操作" width="200" align="center" fixed="right">
              <template #default="scope">
                <el-button
                  text
                  :icon="Edit"
                  @click="onOpenDialog(FormOperation.edit, scope.row)"
                  v-permiss="getPermissKey(routeHook.ViewFeature.edit)"
                >
                  编辑
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
          <Pagination
            :option="table_module.query"
            :total="table_module.pageTotal"
            @current-page-change="getData"
            @size-chage="getData"
          />
        </div>
      </el-footer>
    </el-container>

    <!--编辑-->
    <modify-diaglog
      :title="table_module.diaglogTitle"
      ref="modifyDiaglogRef"
      @saved="getData"
    ></modify-diaglog>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive, onMounted } from 'vue'
import {
  ElTag,
  ElContainer,
  ElButton,
  ElInput,
  ElMain,
  ElHeader,
  ElTable,
  ElTableColumn,
  ElScrollbar,
  ElFooter,
  ElIcon
} from 'element-plus'
import { Delete, Edit, Search, Plus } from '@element-plus/icons-vue'
import modifyDiaglog, { type Item } from '../components/biz/modifySubMenu'

import {
  EnumSelect,
  showLoading,
  closeLoading,
  type IPageParam,
  type Table_Module_Base,
  FormOperation
} from 'co6co'
import { routeHook, deleteHook, useDictHook } from 'co6co-right'
import { get_store } from 'co6co-wx'
import * as api from '../api/biz/wxsubmenu'
import { DictTypeCodes } from '../api/app'

interface IQueryItem extends IPageParam {
  name?: string
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
const store = get_store()
const { selectData, getName, getTagType } = useDictHook.useState()
const dictHook = useDictHook.useDictSelect()

// 获取表格数据
const getData = () => {
  showLoading()
  api
    .get_table_svc(table_module.query)
    .then((res) => {
      table_module.data = res.data
      table_module.pageTotal = res.total || -1
    })
    .finally(() => {
      closeLoading()
    })
}

onMounted(async () => {
  await dictHook.queryByCode(DictTypeCodes.SubMenu)
  getData()
})
// 查询操作
const onSearch = () => {
  table_module.query.pageIndex = 1
  getData()
}

//增加/修改
const modifyDiaglogRef = ref<InstanceType<typeof modifyDiaglog>>()
const onOpenDialog = (operation: FormOperation, row?: Item) => {
  table_module.diaglogTitle = operation == FormOperation.add ? '增加地点' : `编辑[${row?.name}]地点`
  table_module.currentItem = row
  modifyDiaglogRef.value?.openDialog(operation, row)
}

// 删除操作
const { deleteSvc } = deleteHook.default(api.del_svc, getData)
const onDelete = (_: number, row: Item) => {
  deleteSvc(row.id, row.name)
}
</script>
