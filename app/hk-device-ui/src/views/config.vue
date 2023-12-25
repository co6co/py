<template>
  <div>
    <div class="container">
      <el-row :gutter="24">
        <div class="handle-box">
          <el-input v-model="table_module.query.name" placeholder="配置名" class="handle-input mr10"></el-input>
          <el-input v-model="table_module.query.code" placeholder="配置code" class="handle-input mr10"></el-input>
  
          <el-link type="primary" title="更多" @click="table_module.moreOption=!table_module.moreOption">
            <ElIcon :size="20">
              <MoreFilled />
            </ElIcon>
          </el-link>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
          <el-button type="primary" :icon="Plus" @click="onOpenDialog(0)">新增</el-button>
        </div>
      </el-row>
      <el-row :gutter="24" v-if="table_module.moreOption">
        <div class="handle-box">
          <div class="el-select mr10">
            <el-link type="info" @click="setDatetime(0,0.5)">0.5h内</el-link>
            <el-link type="info" @click="setDatetime(0,1)">1h内</el-link>
            <el-link type="info" @click="setDatetime(1,24)">今天</el-link>
            <el-date-picker style="margin-top: 3px;" v-model="table_module.query.datetimes" format="YYYY-MM-DD HH:mm:ss"
              value-format="YYYY-MM-DD HH:mm:ss" type="datetimerange" range-separator="至" start-placeholder="开始时间"
              end-placeholder="结束时间" title="创建时间" />
          </div>
        </div>
      </el-row>
      <el-row :gutter="24">
        <el-table highlight-current-row @sort-change="onColChange" :row-class-name="tableRowProp"
          :data="table_module.data" border class="table" ref="tableInstance" @row-click="onTableSelect"
          header-cell-class-name="table-header">
          <el-table-column prop="id" label="ID" width="80" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
  
          <el-table-column prop="name" label="名称" width="120" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="code" label="名称" width="130" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column width="160" prop="createTime" label="创建时间" sortable
            :show-overflow-tooltip="true"></el-table-column>
  
          <el-table-column width="160" prop="updateTime" label="更新时间" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column label="操作" width="316" align="center">
            <template #default="scope">
              <el-button text :icon="Edit" @click="onOpenDialog(1,scope.row)" v-if="scope.row.deviceType!='1'">
                修改
              </el-button>
              <el-button text :icon="Delete" @click="onDelete(scope.row)" v-if="scope.row.deviceType!='1'">
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
        <div class="pagination">
          <el-pagination background layout="prev, pager, next,total,jumper" :current-page="table_module.query.pageIndex"
            :page-sizes="[100, 200, 300, 400]" :page-size="table_module.query.pageSize" :total="table_module.pageTotal"
            @current-change="onPageChange">
          </el-pagination>
        </div>
  
      </el-row>
    </div>
    <!-- 弹出框 -->
    <el-dialog :title="form.title" v-model="form.dialogVisible" style="width:40%; height:50%;" @keydown.ctrl="keyDown">
      <el-form label-width="90px" ref="dialogForm" :rules="rules" :model="form.fromData" style="max-width: 360px">
  
        <el-form-item label="名称" prop="name">
          <el-input v-model="form.fromData.name" placeholder="配置名称"></el-input>
        </el-form-item>
        <el-form-item label="代码" prop="code">
          <el-input v-model="form.fromData.code" placeholder="配置代码"></el-input>
        </el-form-item>
        <el-form-item label="配置值" prop="value">
          <el-input v-model="form.fromData.value" placeholder="配置值"></el-input>
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
import { ref, watch, reactive, nextTick, PropType, onMounted, onBeforeUnmount, computed } from 'vue';
import { ElMessage, ElMessageBox, FormRules, FormInstance, ElTreeSelect, dayjs } from 'element-plus';
import { TreeNode } from 'element-plus/es/components/tree-v2/src/types'
import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import { Delete, Edit, Search, Compass, MoreFilled, Download, Plus, Minus } from '@element-plus/icons-vue';
import * as api from '../api/config';
import * as res_api from '../api';
import { detailsInfo } from '../components/details';
import { imgVideo, types } from '../components/player';
import { str2Obj, createStateEndDatetime } from '../utils'

interface TableRow {
  id: number;
  code: string;
  name: string;
  createTime: string;
  updateTime: string;
  value: string;
}
interface Query extends IpageParam {
  name: string;
  code: string;
  datetimes: Array<string>,
}
interface table_module {
  query: Query;
  moreOption: boolean;
  data: TableRow[];
  currentRow?: TableRow;
  pageTotal: number;
}

const tableInstance = ref<any>(null);
const currentTableItemIndex = ref<number>();
const table_module = reactive<table_module>({
  query: {
    name: "",
    code: "",
    datetimes: [],
    pageIndex: 1,
    pageSize: 10,
    order: "asc",
    orderBy: ""
  },
  moreOption: false,
  data: [],
  pageTotal: 0
});
const setDatetime = (t: number, i: number) => {
  table_module.query.datetimes = createStateEndDatetime(t, i)
}

// 排序
const onColChange = (column: any) => {
  table_module.query.order = column.order === 'descending' ? 'desc' : 'asc'
  table_module.query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}
const tableRowProp = (data: { row: any, rowIndex: number }) => {
  data.row.index = data.rowIndex;
}
const onRefesh = () => {
  getData();
}
// 查询操作
const onSearch = () => {
  getData();
};
const getQuery = () => {
  table_module.query.pageIndex = table_module.query.pageIndex || 1;
}
// 获取表格数据
const getData = () => {
  getQuery(), api.list_svc(table_module.query).then(res => {
    if (res.code == 0) {
      table_module.data = res.data;
      table_module.pageTotal = res.total || 0;
    } else {
      ElMessage.error(res.message);
    }
  });
};
getData();
// 分页导航
const onPageChange = (pageIndex: number) => {
  let totalPage = Math.ceil(table_module.pageTotal / table_module.query.pageSize.valueOf())
  if (pageIndex < 1) ElMessage.error("已经是第一页了");
  else if (pageIndex > totalPage) ElMessage.error("已经是最后一页了");
  else table_module.query.pageIndex = pageIndex, getData();
};
const setTableSelectItem = (index: number) => {
  if (tableInstance._value.data && index > -1 && index < tableInstance._value.data.length) {
    let row = tableInstance._value.data[index]
    tableInstance._value.setCurrentRow(row);
    onTableSelect(row)
  }
}
const onTableSelect = (row: any) => {
  currentTableItemIndex.value = row.index;
  table_module.currentRow = row;
}
const keyDown = (e: KeyboardEvent) => {
  if (e.ctrlKey) {
    if (['ArrowLeft', 'ArrowRight'].indexOf(e.key) > -1) {
      let current = table_module.query.pageIndex.valueOf();
      let v = e.key == 'ArrowRight' || e.key == 'd' ? (current + 1) : (current - 1);
      onPageChange(v)
    }
    if (['ArrowUp', 'ArrowDown'].indexOf(e.key) > -1) {
      let current = currentTableItemIndex.value;
      if (!current) current = 0;
      let v = e.key == 'ArrowDown' || e.key == 's' ? (current + 1) : (current - 1);
      if (0 <= v && v < tableInstance._value.data.length) {
        setTableSelectItem(v)
      } else {
        if (v < 0) ElMessage.error("已经是第一条了");
        else if (v >= tableInstance._value.data.length) ElMessage.error("已经是最后一条了");
      }
    }
  }
  //process_view.value.keyDown(e) 
  e.stopPropagation()
}
//**详细信息 */
interface FromData {
  deviceType?: number;
  name: string;
  code: string;
  value: string;
}
interface dialogDataType {
  dialogVisible: boolean,
  operation: 0 | 1 | number,
  title?: string;
  id: number
  fromData: FromData
}
let dialogData = {
  dialogVisible: false,
  operation: 0,
  id: 0,
  fromData: {
    code: "",
    name: "",
    value: ""
  }
}
const dialogForm = ref<FormInstance>();
const rules: FormRules = {
  code: [{ required: true, message: '配置code', trigger: ['blur'] }],
  name: [{ required: true, message: '请输入设备名称', trigger: 'blur' }],
  value: [{ required: true, message: '请输入配置值', trigger: ['blur'] }]
};
let form = reactive<dialogDataType>(dialogData);
const onOpenDialog = (operation: 0 | 1, row?: any) => {
  form.dialogVisible = true
  table_module.currentRow = row

  form.dialogVisible = true
  form.operation = operation
  form.id = -1;
  switch (operation) {
    case 0:
      form.title = "增加";
      form.fromData.code = ""
      form.fromData.name = ""
      form.fromData.value = ""
      break;
    case 1:
      form.id = row.id
      form.title = "编辑";
      form.fromData.value = row.value
      form.fromData.code = row.code
      form.fromData.name = row.name
      break;
  }
}
const onDialogSave = (formEl: FormInstance | undefined) => {
  if (!formEl) return;
  formEl.validate(value => {
    if (value) {
      if (form.operation == 0) {
        api.add_config_svc(form.fromData).then(res => {
          if (res.code == 0) {
            form.dialogVisible = false;
            ElMessage.success(`增加成功`);
            getData()
          }
          else { ElMessage.error(`增加失败:${res.message}`); }
        })
      } else {
        api.edit_config_svc(form.id, form.fromData).then(res => {
          if (res.code == 0) {
            form.dialogVisible = false;
            ElMessage.success(`编辑成功`);
            getData()
          }
          else { ElMessage.error(`编辑失败:${res.message}`); }
        })
      }
    } else {
      ElMessage.error("请检查输入的数据！")
      return false;
    }
  })
}

// 删除操作
const onDelete = (row: TableRow) => {
  // 二次确认删除
  ElMessageBox.confirm(`确定要删除"${row.name}\t${row.code}"配置吗？`, '提示', {
    type: 'warning'
  })
    .then(() => {
      api.del_config_svc(row.id).then(res => {
        if (res.code == 0) ElMessage.success('删除成功'), getData();
        else ElMessage.error(`删除失败:${res.message}`);
      }).finally(() => {
      })
    })
    .catch(() => { });
};
//**end 打标签 */
</script> 
<style  scoped lang="less">
.el-link {
  margin-right: 8px;
}

.el-link .el-icon--right.el-icon {
  vertical-align: text-bottom;
}

.handle-box {
  margin: 3px 0;
}

.handle-select {
  width: 120px;
}

.handle-input {
  width: 300px;
}

.table {
  width: 100%;
  font-size: 14px;
}

.red {
  color: #F56C6C;
}

.mr10 {
  margin-right: 10px;
}

.table-td-thumb {
  display: block;
  margin: auto;
  width: 40px;
  height: 40px;
}

.view .title {
  color: var(--el-text-color-regular);
  font-size: 18px;
  margin: 10px 0;
}

.view .value {
  color: var(--el-text-color-primary);
  font-size: 16px;
  margin: 10px 0;
}

::v-deep .view .radius {
  height: 40px;
  width: 70%;
  border: 1px solid var(--el-border-color);
  border-radius: 0;
  margin-top: 20px;
}

::v-deep .el-table tr,
.el-table__row {
  cursor: pointer;
}

.formItem {
  display: flex;
  align-items: center;
  display: inline-block;

  .label {
    display: inline-block;
    color: #aaa;
    padding: 0 5px;
  }
}

::v-deep .el-dialog__body {
  height: 70%;
  overflow: auto;
}

.menuInfo {
  .el-menu {
    width: auto;

    .el-menu-item {
      padding: 10px;
      height: 40px;
    }
  }
}

::v-deep .streamInfo {
  max-width: 560px;

  .el-card {
    margin: 2px 0;
  }

  .el-form-item {
    padding: 8px 0;
  }

  .el-form-item__content {
    width: 470px;
  }

  .el-card__body {
    padding: 2px 5px;
  }

  .el-form-item__label {
    width: 70px;
  }

}
</style>
