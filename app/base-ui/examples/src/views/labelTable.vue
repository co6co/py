<!-- eslint-disable camelcase -->
<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-input v-model="query.name" placeholder="名称" class="handle-input mr10" />
          <el-input v-model="query.alias" placeholder="别名" class="handle-input mr10" />

          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
          <el-button
            type="primary"
            v-permiss="getPermissKey(ViewFeature.add)"
            :icon="Plus"
            @click="onOpenDialog(0)"
            >新增</el-button
          >
        </div>
      </el-header>
      <el-main>
        <el-scrollbar>
          <el-table
            highlight-current-row
            @sort-change="onColChange"
            :data="tableData"
            border
            class="table"
            ref="multipleTable"
            header-cell-class-name="table-header"
          >
            <el-table-column label="序号" width="55" align="center">
              <template #default="scope"> {{ scope.$index }} </template>
            </el-table-column>
            <el-table-column prop="id" label="ID" width="90" align="center" />
            <el-table-column
              prop="name"
              label="名称"
              sortable="custom"
              :show-overflow-tooltip="true"
            />
            <el-table-column
              prop="alias"
              label="别名"
              sortable="custom"
              :show-overflow-tooltip="true"
            />
            <el-table-column
              prop="order"
              label="排序"
              sortable="custom"
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
            <el-table-column label="操作" width="316" align="center">
              <template #default="scope">
                <el-button
                  text
                  v-permiss="getPermissKey(ViewFeature.edit)"
                  :icon="Edit"
                  @click="onOpenDialog(1, scope.row)"
                >
                  编辑
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
            :current-page="query.pageIndex"
            :page-size="query.pageSize"
            :total="pageTotal"
            @current-change="onCurrentPageChange"
          />
        </div>
      </el-footer>
    </el-container>

    <!-- 弹出框 -->
    <el-dialog :title="form.title" v-model="form.dialogVisible" width="30%">
      <el-form label-width="70px" ref="dialogForm" :rules="rules" :model="form.fromData">
        <el-form-item label="名称" prop="name">
          <el-input v-model="form.fromData.name" placeholder="英文字符" />
        </el-form-item>
        <el-form-item label="别名" prop="alias">
          <el-input v-model="form.fromData.alias" placeholder="中文字符" />
        </el-form-item>
        <el-form-item label="排序" prop="order">
          <el-input-number v-model="form.fromData.order" placeholder="排序" />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="form.dialogVisible = false">关闭</el-button>
          <el-button @click="onDialogSave(dialogForm)">保存</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue'
import { ElMessage, ElMessageBox, type FormRules, type FormInstance } from 'element-plus'
import { Delete, Edit, Search, Compass, Plus, Download } from '@element-plus/icons-vue'

// eslint-disable-next-line camelcase
import { get_list_svc, add_svc, edit_svc, del_svc, get_exist_svc } from '../api/label'
import { showLoading, closeLoading } from '../components/Logining'
import { usePermission, ViewFeature } from '../hook/sys/useRoute'

const { getPermissKey } = usePermission()
interface TableItem {
  id: number
  name: string
  alias: string
  order: number
}
interface QueryType {
  name?: string
  alias?: string
  pageIndex: number
  pageSize: number
  order?: string
  orderBy?: string
}
const query = reactive<QueryType>({
  name: '',
  alias: undefined,
  pageIndex: 1,
  pageSize: 10,
  order: ''
})
const tableData = ref<TableItem[]>([])
const pageTotal = ref(0)
// 获取表格数据
const getData = () => {
  showLoading()
  get_list_svc(query)
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
const onColChange = (column: any) => {
  //console.info(column)
  query.order = column.order === 'descending' ? 'desc' : 'asc'
  query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
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
        // eslint-disable-next-line @typescript-eslint/no-empty-function
        .finally(() => {})
    })
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    .catch(() => {})
}
const validName = (rule: any, value: string, back: Function) => {
  console.info(rule)
  if (value.length < 2) return (rule.message = '长度应该大于2'), back(new Error(rule.message))
  get_exist_svc(dialogData.id, value).then((res) => {
    if (res.code == 0) return (rule.message = res.message), back(new Error(rule.message))
    else return back()
  })
}
//弹出框 add and edit
const dialogForm = ref<FormInstance>()
const rules: FormRules = {
  name: [
    {
      required: true,
      pattern: '[0-9A-Za-z_+=-]{1,}',
      validator: validName,
      message: '请输入名称',
      trigger: ['blur']
    }
  ],
  alias: [{ required: true, message: '请输入别名', trigger: 'blur' }]
}
let dialogData = {
  dialogVisible: false,
  title: '',
  operation: 0,
  id: -1,
  fromData: {
    name: '',
    alias: '',
    order: '',
    type: -1
  }
}
let form = reactive(dialogData)
const onOpenDialog = (type: number, row?: any) => {
  form.dialogVisible = true
  form.operation = type
  form.id = -1
  switch (type) {
    case 0:
      form.title = '增加'
      form.fromData.name = ''
      form.fromData.alias = ''
      form.fromData.type = 0
      break
    case 1:
      form.id = row.id
      form.title = '编辑'
      form.fromData.name = row.name
      form.fromData.alias = row.alias
      form.fromData.order = row.order
      form.fromData.type = 0
      break
  }
}
const onDialogSave = (formEl: FormInstance | undefined) => {
  if (!formEl) return
  formEl.validate((value) => {
    if (value) {
      if (form.operation == 0) {
        add_svc(form.fromData).then((res) => {
          if (res.code == 0) {
            form.dialogVisible = false
            ElMessage.success(`增加标签成功`)
            getData()
          } else {
            ElMessage.error(`增加标签失败:${res.message}`)
          }
        })
      } else {
        edit_svc(form.id, form.fromData).then((res) => {
          if (res.code == 0) {
            form.dialogVisible = false
            ElMessage.success(`编辑标签成功`)
            getData()
          } else {
            ElMessage.error(`编辑标签失败:${res.message}`)
          }
        })
      }
    } else {
      ElMessage.error('请检查输入的数据！')
      return false
    }
  })
}
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
