<template>
  <div>
    <div class="container">
      <el-row :gutter="24">
        <div class="handle-box">
          <el-input v-model="table_module.query.name" placeholder="用户名" class="handle-input mr10"></el-input>
          <el-select style="width: 160px" class="mr10" clearable v-model="table_module.query.category" placeholder="设备类型">
            <el-option v-for="item  in DeviceCategoryRef?.categoryList" :key="item.uid" :label="item.key" :value="item.value" />
          </el-select>
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
          <el-table-column prop="uuid" label="UUID" width="120" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="name" label="名称" width="80" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="innerIp" label="内部地址" width="120" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="ip" label="网络地址" width="120" align="center" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column prop="deviceType" label="设备类型" width="120" sortable :show-overflow-tooltip="true">
            <template #default="scope">
              <el-tag> {{table_module.getDeviceName(scope.row.deviceType)}} </el-tag>
            </template>
          </el-table-column>
  
          <el-table-column width="160" prop="createTime" label="创建时间" sortable
            :show-overflow-tooltip="true"></el-table-column>
          <el-table-column label="操作" width="316" align="center">
            <template #default="scope">
              <el-button text :icon="Edit" @click="onOpenDialog(1,scope.row)"  v-if="scope.row.deviceType!='1'">
                修改
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
    <el-dialog :title="form.title" v-model="form.dialogVisible" style="width:50%; height:80%;" @keydown.ctrl="keyDown">
      <el-form label-width="90px" ref="dialogForm" :rules="rules" :model="form.fromData" style="max-width: 460px">
        <el-form-item label="设备类型" prop="deviceType">
          <el-select style="width:160px" class="mr10" clearable v-model="form.fromData.deviceType" placeholder="请选择">
            <el-option v-for="item in DeviceCategoryRef?.categoryList" :disabled="item.value==DeviceCategoryRef?.boxCategory" :key="item.uid" :label="item.key"
              :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="名称" prop="name">
          <el-input v-model="form.fromData.name" placeholder="设备名称"></el-input>
        </el-form-item>
        <el-form-item label="内网IP" prop="innerIp">
          <el-input v-model="form.fromData.innerIp" placeholder="内外IP"></el-input>
        </el-form-item>
  
        <el-form-item label="流信息" class="streamInfo">
          <el-card v-for="(steam,index) in form.fromData.streamUrls" :key="index">
            <el-form-item label="流名称" :prop="'streamUrls.' + index + '.name'" :rules="{
                required: true,
                message: '请输入流名称',
                trigger: 'blur',
              }">
              <el-input v-model="steam.name" placeholder="视频流名称" required></el-input>
            </el-form-item>
  
            <el-form-item label="流地址" :prop="'streamUrls.' + index + '.url'" :rules="{
                required: true,
                message: '请输入流地址',
                trigger: 'blur',
              }">
              <el-input v-model="steam.url" placeholder="视频流地址" required></el-input>
            </el-form-item>
            <el-button @click="removeStream" :icon="Minus"> </el-button>
          </el-card>
          <el-button @click="addStream" :icon="Plus"></el-button>
        </el-form-item>
  
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="form.dialogVisible = false">关闭</el-button>
          <el-button @click="onDialogSave(dialogForm)">保存</el-button>
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
import { Delete, Edit, Search, Compass, MoreFilled, Download, Plus, Minus } from '@element-plus/icons-vue';
import * as api from '../api/device';
import * as res_api from '../api';
import * as t from '../store/types/devices'
import { detailsInfo } from '../components/details';
import { imgVideo, types } from '../components/player';
import { str2Obj,createStateEndDatetime } from '../utils'



 
interface TableRow {
  id: number;
  uuid: string;
  deviceType: number;
  innerIp: string;
  ip: string;
  name: string; 
  createTime: string;
  poster?: string; 
  streams?: string;
}
interface Query extends IpageParam {
  name: string;
  category?: number;
  datetimes: Array<string>,
}
interface table_module {
  query: Query;
  moreOption: boolean;
  data: TableRow[];
  currentRow?: TableRow;
  pageTotal: number;
  getDeviceName: (category?: number) => String
}

const tableInstance = ref<any>(null);
const currentTableItemIndex = ref<number>();
const table_module = reactive<table_module>({
  query: {
    name: "",
    datetimes: [],
    pageIndex: 1,
    pageSize: 10,
    order: "asc",
    orderBy: ""
  },
  moreOption: false,
  data: [],
  pageTotal: -1,
  getDeviceName: (value?: number) => {
    if (DeviceCategoryRef.value && DeviceCategoryRef.value.categoryList.length > 0) {
      let result = value == null ? "" : DeviceCategoryRef.value.categoryList.find(m => m.value == value)?.key
      if (result == null) return "" 
      return result
    } 
    return ""
  }
}); 
const setDatetime = (t: number, i: number) => { 
  table_module.query.datetimes =createStateEndDatetime(t,i)
}

interface DeviceCategory{
  categoryList:Array<EnumType>
  cameraCategory:number
  boxCategory:number
}

const DeviceCategoryRef = ref<DeviceCategory>( )
const getDeviceType = async () => {
  const res = await api.dev_type_svc()
  if (res.code == 0) { 
    DeviceCategoryRef.value = res.data
  }
}
getDeviceType()
onMounted(() => {

})
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
  getQuery(), api.dev_list_svc(table_module.query).then(res => {
    if (res.code == 0) {
      table_module.data = res.data;
      table_module.pageTotal = res.total || -1;
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
  innerIp: string;
  name: string;
  streamUrls: Array<{ name: String, url: String }>
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
    innerIp: "",
    name: "",
    streamUrls: []
  }
}
const dialogForm = ref<FormInstance>();
const rules: FormRules = {
  deviceType: [{ required: true, message: '请选择设备类型', trigger: ['blur'] }],
  name: [{ required: true, message: '请输入设备名称', trigger: 'blur' }],
  innerIp: [{ required: true, message: '请输入设备IP', trigger: ['blur'] }],
  streamName: [{ required: true, message: '请视频地址名称', trigger: ['blur'] }],
  streamUrl: [{ required: true, message: '请视频地址', trigger: ['blur'] }],
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
      form.fromData.innerIp = ""
      form.fromData.name = ""
      break;
    case 1:
      form.id = row.id
      form.title = "编辑";
      form.fromData.deviceType = row.deviceType
      form.fromData.innerIp = row.innerIp
      form.fromData.name = row.name
      if (row.streams && typeof row.streams == 'string')form.fromData.streamUrls=JSON.parse(row.streams); 
      break;
  }
}
const removeStream = (index: number) => {
  form.fromData.streamUrls.splice(index, 1)
}
const addStream = () => {
  form.fromData.streamUrls.push({ name: "", url: "" })
}
const onDialogSave = (formEl: FormInstance | undefined) => {
  if (!formEl) return;
  formEl.validate(value => {
    if (value) {
      if (form.operation == 0) {
        api.add_camera_svc(form.fromData).then(res => {
          if (res.code == 0) {
            form.dialogVisible = false;
            ElMessage.success(`增加成功`);
            getData()
          }
          else { ElMessage.error(`增加失败:${res.message}`); }
        })
      } else {
        api.edit_camera_svc(form.id, form.fromData).then(res => {
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
//**视频下信息 */
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
