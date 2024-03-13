<template>
	<div>
		<div class="container"> 
				<el-row :gutter="24"> 
						<div class="handle-box"> 
							<el-select style="width:160px"  class="mr10"  v-model="table_module.query.alarmType" placeholder="请选择">
							<!--	<el-option 
									v-for="item  in form_attach_data.flow_status" 
									:key="item.key" :label="item.label" :value="item.value"
								/> -->
								<el-option label="123456"></el-option>
							</el-select> 
							<el-link type="primary" title="更多" @click="table_module.moreOption=!table_module.moreOption"><ElIcon :size="20"><MoreFilled /></ElIcon></el-link>
							<el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>  
						</div> 
				</el-row>
				<el-row :gutter="24" v-if="table_module.moreOption"> 
					<div class="handle-box">  
						<div class="el-select mr10">
							<el-link type="info" @click="setDatetime(0,0.5)">0.5h内</el-link>
							<el-link type="info"  @click="setDatetime(0,1)">1h内</el-link> 
							<el-link type="info"  @click="setDatetime(1,24)">今天</el-link> 
							<el-date-picker  style="margin-top: 3px;" 
								v-model="table_module.query.datetimes" 
								format="YYYY-MM-DD HH:mm:ss"
								value-format="YYYY-MM-DD HH:mm:ss"
								type="datetimerange"
								range-separator="至"
								start-placeholder="开始时间"
								end-placeholder="结束时间"
								title="告警事件"
							/>
						</div> 
					</div>
				</el-row>
				<el-row :gutter="24">  
						<el-table highlight-current-row @sort-change="onColChange"  
						:row-class-name="tableRowProp"
						:data="table_module.data" border class="table" ref="tableInstance" @row-click="onTableSelect" header-cell-class-name="table-header">  
							<el-table-column prop="uuid"  label="ID" width="80" align="center" sortable :show-overflow-tooltip="true"></el-table-column>
							<el-table-column prop="alarmType" label="告警类型"  width="119" sortable  :show-overflow-tooltip="true"></el-table-column>
							<el-table-column prop="alarmTypePO.desc" label="告警描述"  width="119" sortable  :show-overflow-tooltip="true"></el-table-column>
							<el-table-column prop="alarmTypePO.desc" label="告警描述"  width="119" sortable  :show-overflow-tooltip="true"></el-table-column>
							<el-table-column label="任务类型" width="110"  sortable prop="flowStatus">
								<template #default="scope"> <el-tag  >{{ scope.row.taskSession }}--{{ scope.row.taskDesc }} </el-tag></template>
							</el-table-column> 
							 
							<el-table-column width="160" prop="alarmTime" label="告警时间" sortable  :show-overflow-tooltip="true"></el-table-column> 
							<el-table-column width="160" prop="createTime" label="入库时间" sortable  :show-overflow-tooltip="true"></el-table-column> 
							<el-table-column label="操作" width="316" align="center">
							<template #default="scope">  
								<el-button text :icon="Edit" @click="onOpenPage('/alarmdetail.html', scope.row)">
									详细信息
								</el-button> 
								<el-button text :icon="Edit" @click="onOpenPage('/alarmpreview.html', scope.row)">
									告警视频
								</el-button> 
							</template>
						</el-table-column>
						</el-table>
						<div class="pagination"> 
							<el-pagination
								background
								layout="prev, pager, next,total,jumper"
								:current-page="table_module.query.pageIndex" 
								:page-sizes="[100, 200, 300, 400]"
								:page-size="table_module.query.pageSize" 
								:total="table_module.pageTotal"
								@current-change="onPageChange">  
							</el-pagination>
						</div> 
					 
				</el-row> 
		</div>
		  
	</div>
</template>

<script setup lang="ts" name="basetable">
import { ref, watch,reactive, nextTick, type  PropType ,onMounted, onBeforeUnmount, computed  } from 'vue';
import { ElMessage, ElMessageBox,type  FormRules,type  FormInstance,ElTreeSelect, dayjs ,ElTable} from 'element-plus';
import { type  TreeNode } from 'element-plus/es/components/tree-v2/src/types'
import {type   TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import { Delete, Edit, Search, Compass,MoreFilled,Download } from '@element-plus/icons-vue'; 
import  * as api from '../api/alarm'; 
import { type AlarmItem, getResources } from '../components/biz'

import  { imgVideo,types} from '../components/player';    
import {useAppDataStore} from '../store/appStore'
import {  createStateEndDatetime } from '../utils';

//页面跳转相关 1.
import { useRouter } from "vue-router";
const router = useRouter()

const dataStore=useAppDataStore() 
const onOpenPage=(path:string,row:any)=>{ 
	dataStore.setState(row)  
	router.push({
		path: path,
		//后面两个参数没用到
		/**
		query: {
			mode: "edit",
		},
		params:{data:"123456"}, 
		 */
	});
}

interface Query extends IpageParam{
  datetimes:Array<string>,
  alarmType:String,  
}
interface table_module  {
    query:Query, 
    moreOption:boolean,
    data:AlarmItem[],
    currentRow?:AlarmItem,
    pageTotal:number,
} 
const elTreeInstance=ref<any>(null) 
const selectedVal = ref<any[]>();
const cacheData = [ { value: 5, label: '位置信息' }]
const form_attach_data={}
const tableInstance=ref<InstanceType< typeof ElTable>>(); 
const currentTableItemIndex = ref<number>();  
const table_module = reactive<table_module>({ 
	query:{
		alarmType: '', 
		datetimes:[], 
		pageIndex: 1,
		pageSize: 15,
		order:'asc',
		orderBy:'', 
	},
	moreOption:false,  
    data:[],
	pageTotal:-1,  
}); 
const setDatetime = (t: number, i: number) => {
		table_module.query.datetimes = createStateEndDatetime(t, i);
};
 
   // 排序
const onColChange = (column:any) =>  {  
	table_module.query.order = column.order  === 'descending' ? 'desc' : 'asc'
	table_module.query.orderBy = column.prop 
	if (column) getData() // 获取数据的方法
} 

 
const tableRowProp=( data:{ row:any, rowIndex :number})=>{ 
	data.row.index=data.rowIndex; 
	return ''
} 
const onRefesh=()=>{
    getData();
}
// 查询操作
const onSearch = () => { 
	getData();
};  
const getQuery=()=>{
	table_module.query.pageIndex = table_module.query.pageIndex||1;  
}
// 获取表格数据
const getData = ( ) => {   
	getQuery(),api.list_svc(table_module.query).then(res => {
        if (res.code==0){
            table_module.data = res.data; 
		    table_module.pageTotal = res.total || -1; 
        }else{
            ElMessage.error(res.message); 
        } 
	});
};
getData();  
// 分页导航
const onPageChange = (pageIndex: number) => {
	let totalPage=Math.ceil(table_module.pageTotal/table_module.query.pageSize.valueOf())
	if(pageIndex<1)  ElMessage.error("已经是第一页了");  
	else if (pageIndex>totalPage)  ElMessage.error("已经是最后一页了"); 
	else table_module.query.pageIndex = pageIndex,getData(); 
}; 
const setTableSelectItem=(index:number)=>{  
	if(tableInstance.value&&tableInstance.value.data&&index>-1 && index< tableInstance.value.data.length ){ 
		let row=tableInstance.value.data[index] 
		tableInstance.value.setCurrentRow(row);  
		onTableSelect(row)
	}
}
const onTableSelect=(row:any)=>{  
	currentTableItemIndex.value=row.index;
	table_module.currentRow=row;
} 
const keyDown=(e:KeyboardEvent)=>{ 
	if(e.ctrlKey){
		if (['ArrowLeft','ArrowRight'].indexOf( e.key) >-1){
			let current=table_module.query.pageIndex.valueOf();
			let v=e.key=='ArrowRight'||e.key=='d'?(current+1):(current-1);
			onPageChange(v)
		}  
		if (['ArrowUp','ArrowDown'].indexOf( e.key) >-1){
			let current=currentTableItemIndex.value;
			if (!current)current=0;
			let v=e.key=='ArrowDown'||e.key=='s'?(current+1):(current-1); 
			if(tableInstance.value&&0<=v && v<tableInstance.value.data.length){ 
				setTableSelectItem(v)
			}else { 
				if(v<0)ElMessage.error("已经是第一条了");  
				else if (tableInstance.value&&v>=tableInstance.value.data.length)  ElMessage.error("已经是最后一条了"); 
			}
		}  
	} 
	//process_view.value.keyDown(e) 
    e.stopPropagation() 
} 
 
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
::v-deep  .el-table tr,.el-table__row {cursor: pointer;  } 

.formItem {
  display: flex;
  align-items: center;
  display: inline-block;
  .label{display: inline-block; color: #aaa; padding: 0 5px;}
}
   
::v-deep .el-dialog__body{height:70%; overflow: auto;}
.menuInfo{.el-menu{width: auto; .el-menu-item{padding: 10px; height: 40px;}}}

/**

::v-deep .el-dialog{
	.el-dialog__header{padding: 5px;}
	.el-dialog__body{padding: 15px 5px;}
	.el-dialog__footer{padding: 5px;}
} */
</style>
