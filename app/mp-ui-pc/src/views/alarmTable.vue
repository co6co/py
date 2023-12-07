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
								<el-button text :icon="Edit" @click="onOpenDialog(scope.row)">
									详细信息
								</el-button> 
								<el-button text :icon="Edit" @click="onOpen2Dialog(scope.row)">
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
		<!-- 弹出框 -->
		<el-dialog title="详细信息" v-model="form.dialogVisible"   style="width:98%; height: 98%;" @keydown.ctrl="keyDown">
			<details-info :data="form.data"  ></details-info>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="form.dialogVisible = false">取 消</el-button> 
				</span>
			</template>
		</el-dialog>

		<!-- 弹出框 -->
		<el-dialog title="详细信息" v-model="form2.dialogVisible"   style="width:98%; height: 90%;" @keydown.ctrl="keyDown">
			<el-row> 
				<el-col :span="12">
					<img-video :viewOption="form2.data"  ></img-video>
				</el-col>
				<el-col :span="12">
					 123
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
import { ref, watch,reactive, nextTick, PropType ,onMounted, onBeforeUnmount, computed  } from 'vue';
import { ElMessage, ElMessageBox,FormRules,FormInstance,ElTreeSelect, dayjs } from 'element-plus';
import { TreeNode } from 'element-plus/es/components/tree-v2/src/types'
import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import { Delete, Edit, Search, Compass,MoreFilled,Download } from '@element-plus/icons-vue'; 
import  * as api from '../api/alarm'; 
import  * as res_api from '../api'; 
import  { detailsInfo} from '../components/details';    
import  { imgVideo,types} from '../components/player';    
import {str2Obj} from '../utils'
interface TableRow {
    id: number,
    uuid:string
    alarmType:string
    videoUid:string,
    rawImageUid:string,
    markedImageUid:string,
    alarmTime:string,
    createTime:string 
}
interface Query extends IpageParam{
  datetimes:Array<string>,
  alarmType:String,  
}
interface table_module  {
    query:Query, 
    moreOption:boolean,
    data:TableRow[],
    currentRow?:TableRow,
    pageTotal:number,
} 
const elTreeInstance=ref<any>(null) 
const selectedVal = ref<any[]>();
const cacheData = [ { value: 5, label: '位置信息' }]
const form_attach_data={}
const tableInstance=ref<any>(null); 
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
const setDatetime=(t:number, i:number)=>{ 
	let endDate=null
	let startDate=null
	switch (t){
		case 0:
			endDate=new Date();
			const times=endDate.getTime()-i*3600*1000
			startDate=new Date(times)
			break
		case 1:
			startDate=new Date( dayjs(new Date()) .format('YYYY/MM/DD'))
			endDate=startDate.getTime()+24*3600*1000-1000
			break
		default:
			startDate=new Date( dayjs(new Date()) .format('YYYY/MM/DD'))
			endDate=startDate.getTime()+24*3600*1000 -1000
			break
	}  
	table_module.query.datetimes=[dayjs(startDate) .format('YYYY-MM-DD HH:mm:ss'),dayjs(endDate) .format('YYYY-MM-DD HH:mm:ss')]
}
 
   // 排序
const onColChange = (column:any) =>  {  
	table_module.query.order = column.order  === 'descending' ? 'desc' : 'asc'
	table_module.query.orderBy = column.prop 
	if (column) getData() // 获取数据的方法
} 

 
const tableRowProp=( data:{ row:any, rowIndex :number})=>{ 
	data.row.index=data.rowIndex; 
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
	if(tableInstance._value.data&&index>-1 && index< tableInstance._value.data.length ){ 
		let row=tableInstance._value.data[index] 
		tableInstance._value.setCurrentRow(row);  
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
			if(0<=v && v<tableInstance._value.data.length){ 
				setTableSelectItem(v)
			}else { 
				if(v<0)ElMessage.error("已经是第一条了");  
				else if (v>=tableInstance._value.data.length)  ElMessage.error("已经是最后一条了"); 
			}
		}  
	} 
	//process_view.value.keyDown(e) 
    e.stopPropagation() 
} 
//**详细信息 */
interface dialogDataType{
	dialogVisible:boolean, 
	data:Array<any>
}
let dialogData={
	dialogVisible:false, 
	data:[]
}
let form = reactive<dialogDataType>(dialogData); 
const onOpenDialog=( row?:any)=>{
	form.dialogVisible = true  
	table_module.currentRow=row   
	form.data=[
		{name:"检测结果信息",data:str2Obj( row.alarmAttachPO.result)},
		{name:"视频流信息",data:str2Obj(row.alarmAttachPO.media)},
		{name:"GPS信息",data:str2Obj(row.alarmAttachPO.gps)}]
}


//**视频下信息 */
interface dialog2DataType{
	dialogVisible:boolean, 
	data:Array<types.resourceOption>
}
let dialog2Data={
	dialogVisible:false, 
	data:[]
}
let form2 = ref<dialog2DataType>(dialog2Data); 
const setVideoResource=(uuid:string,option:types.videoOption)=>{
	res_api.request_resource_svc(import.meta.env.VITE_BASE_URL+`/api/resource/poster/${uuid}`).then(res=>{option.poster=res}).catch(e=>option.poster="" );  
	res_api.request_resource_svc(import.meta.env.VITE_BASE_URL+`/api/resource/${uuid}`).then(res=>{ option.url=res}).catch(e=>option.url="" );   
}
const setImageResource=(uuid:string,option:types.imageOption)=>{
	res_api.request_resource_svc(import.meta.env.VITE_BASE_URL+`/api/resource/${uuid}`).then(res=>{ option.url=res}).catch(e=>option.url="" );  
}
const getResultUrl=(uuid:string,isposter:boolean=false)=>{
	if (isposter) return import.meta.env.VITE_BASE_URL+`/api/resource/poster/${uuid}/700/600`;
	return import.meta.env.VITE_BASE_URL+`/api/resource/${uuid}`
}
const onOpen2Dialog=( row:TableRow)=>{
	form2.value.dialogVisible = true  
	table_module.currentRow=row  
	form2.value.data=[  
		{ url: getResultUrl(row.rawImageUid), name:"原始图片" ,poster:getResultUrl(row.rawImageUid,true),type:1},
		{ url:getResultUrl(row.markedImageUid),  name:"标注图片",poster:getResultUrl(row.markedImageUid,true), type:1} ,
	 	{ url:getResultUrl(row.videoUid),  name:"原始视频", poster:getResultUrl(row.videoUid,true),type:0},
	]
	/**
	setImageResource(row.rawImageUid,opt1)
	setImageResource(row.markedImageUid,opt2)
	setVideoResource(row.videoUid,opt3)
	form2.value.data=[opt1,opt2,opt3] 
	*/ 
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
