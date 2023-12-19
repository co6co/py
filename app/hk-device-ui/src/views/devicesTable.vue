<template>
  <div>
    <div class="container">
      <el-row :gutter="24">
        <div class="handle-box">
          <el-input v-model="table_module.query.name" placeholder="设备名称" class="handle-input mr10"></el-input>
          <el-select style="width:160px" class="mr10" v-model="table_module.query.category" placeholder="设备类别">
            <el-option v-for="item  in category_list" :key="item.id" :label="item.name" :value="item.id" />
          </el-select>
          <el-link type="primary" title="更多" @click="table_module.moreOption=!table_module.moreOption">
            <ElIcon :size="20">
              <MoreFilled />
            </ElIcon>
          </el-link>
          <el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
          <el-button type="primary" :icon="Search" @click="onSetting">补光灯设置</el-button>
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
              end-placeholder="结束时间" title="告警事件" />
          </div>
        </div>
      </el-row>
  
      <el-row :gutter="24">
        <el-table highlight-current-row @sort-change="onColChange" :row-class-name="tableRowProp"
          :data="table_module.data" border class="table" ref="tableInstance" @row-click="onTableSelect"
          header-cell-class-name="table-header">
          <el-table-column prop="id" label="ID" width="80" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="ip" label="IP地址" width="119" sortable :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="name" label="名称" width="119" sortable :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="code" label="代码" width="119" sortable :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="categoryName" label="类别名称" width="119" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="categoryCode" label="类别代码" width="119" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column width="160" prop="createTime" label="入库时间" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column label="操作" width="316" align="center">
            <template #default="scope">
              <el-button text :icon="Edit" @click="onOpen2Dialog(scope.row)">
                查看日志
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
    <el-dialog title="设置补光灯" v-model="form.dialogVisible" style="width:98%; height: 98%;" @keydown.ctrl="keyDown">
      <el-form label-width="70px">
        <el-form-item label="开/关">
          <el-switch v-model="form.allows" class="ml-2"
            style="--el-switch-on-color: #13ce66; --el-switch-off-color: #ff4949" />
        </el-form-item>
        <el-form-item label="开始时间">
          <el-time-select v-model="form.startTime" start="18:00" step="00:05" end="21:30" placeholder="选择时间" />
        </el-form-item>
        <el-form-item label="结束时间">
          <el-time-select v-model="form.endTime" start="05:30" step="00:05" end="08:30" placeholder="选择时间" />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="form.dialogVisible = false">取 消</el-button>
          <el-button type="primary" @click="onSave">确 定</el-button>
        </span>
      </template>
    </el-dialog>
  
    <!-- 弹出框 -->
    <el-dialog title="详细信息" v-model="form2.dialogVisible" style="width:98%; height: 90%;" @keydown.ctrl="keyDown">
      <el-row>
        <el-col :span="12">
          <img-video :viewOption="form2.data"></img-video>
        </el-col>
        <el-col :span="12">
        </el-col>
      </el-row>
  
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="form2.dialogVisible = false">取 消</el-button>
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
import { Delete, Edit, Search, Compass, MoreFilled, Download } from '@element-plus/icons-vue';
import * as api from '../api/device';
import * as res_api from '../api';
import { detailsInfo } from '../components/details';
import { imgVideo, types } from '../components/player';
import { str2Obj } from '../utils'

//table Row
interface TableRow {
  id: number,
  ip: string
  categoryName: string
  vendor: number,
  name: string,
  code: string,
  createTime: string,
  updateTime: string,
}
//
interface Query extends IpageParam {
  datetimes: Array<string>,
  name: String,
  category?: number,
}
interface table_module {
  query: Query,
  moreOption: boolean,
  data: TableRow[],
  currentRow?: TableRow,
  pageTotal: number,
}
interface Category {
  id: number;
  name: string;
  code: string;
}
const category_list = ref<Array<Category>>(<Category[]>[])
onMounted(async () => {
  const res = await api.category_list_svc()
  if (res.code == 0) {
    category_list.value = res.data
  }
})

const tableInstance = ref<any>(null);
const currentTableItemIndex = ref<number>();
const table_module = reactive<table_module>({
  query: {
    name: '',
    datetimes: [],
    pageIndex: 1,
    pageSize: 15,
    order: 'asc',
    orderBy: '',
  },
  moreOption: false,
  data: [],
  pageTotal: 0,
});
const setDatetime = (t: number, i: number) => {
  let endDate = null
  let startDate = null
  switch (t) {
    case 0:
      endDate = new Date();
      const times = endDate.getTime() - i * 3600 * 1000
      startDate = new Date(times)
      break
    case 1:
      startDate = new Date(dayjs(new Date()).format('YYYY/MM/DD'))
      endDate = startDate.getTime() + 24 * 3600 * 1000 - 1000
      break
    default:
      startDate = new Date(dayjs(new Date()).format('YYYY/MM/DD'))
      endDate = startDate.getTime() + 24 * 3600 * 1000 - 1000
      break
  }
  table_module.query.datetimes = [dayjs(startDate).format('YYYY-MM-DD HH:mm:ss'), dayjs(endDate).format('YYYY-MM-DD HH:mm:ss')]
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

// 获取表格数据
const getData = () => {
  table_module.query.pageIndex = table_module.query.pageIndex || 1;
  api.list_svc(table_module.query).then(res => {
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
//**补光灯设置 */
interface dialogDataType {
  dialogVisible: boolean;
  allows?: boolean;
  startTime: string;
  endTime: string;
  query?: Query;
}
let dialogData = {
  dialogVisible: false,
  allows: true,
  startTime: "20:00",
  endTime: "07:00"

}
let form = reactive<dialogDataType>(dialogData);

const onSetting = () => {
  form.dialogVisible = true
}
const onSave = () => {
  ElMessageBox.confirm(`确定要设置吗？`, '提示', {
    type: 'warning'
  })
    .then(() => {
      form.query = table_module.query
      api.set_ligth_svc(form).then((res) => {
        if (res.code == 0) ElMessage.success(res.message);
        else ElMessage.error(res.message);
      })
    });
}
//**日志查看 */
interface dialog2DataType {
  dialogVisible: boolean,
  data: Array<types.resourceOption>
}
let dialog2Data = {
  dialogVisible: false,
  data: []
}
let form2 = ref<dialog2DataType>(dialog2Data);
const onOpen2Dialog = (row: TableRow) => {
  form2.value.dialogVisible = true
  table_module.currentRow = row
  form2.value.data = []
}

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
</style>
