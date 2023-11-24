<template>
	<div>
		<div class="container"> 
				<el-row :gutter="24"> 
						<div class="handle-box">  
							<div class="el-select mr10">   
								<!--	<select-tree @load-data="onLoadData" @node-check="onNodeCheck" :key-props="table_module.treeDataMap" :options="table_module.treeData" v-model="table_module.query.positions"   placeholder="位置信息" />-->
							 	<el-tree-select  
								 	ref="elTreeInstance"  
									v-model="selectedVal"  
									lazy
									:load="onLoadTree"  
									:props="table_module.treeDataMap"
									:render-after-expand="false" 
									show-checkbox
									@check="Oncheck"
									:cache-data="cacheData"  
								/>  
							</div>
							<el-input  style="width:160px"   v-model="table_module.query.boatName" placeholder="船名" class="handle-input mr10"></el-input>
							<el-select style="width:160px"  class="mr10"  v-model="table_module.query.flowStatus" placeholder="请选择">
								<el-option 
									v-for="item  in form_attach_data.flow_status" 
									:key="item.key" :label="item.label" :value="item.value"
								/> 
							</el-select> 
							<el-select style="width:160px"  class="mr10"   v-model="table_module.query.manualAuditStatus" placeholder="请选择">
								<el-option 
									v-for="item  in form_attach_data.manual_audit_state" 
									:key="item.key" :label="item.label" :value="item.value"
								/> 
							</el-select>   
 
							<el-link type="primary" title="更多" @click="table_module.queryMoreOption=!table_module.queryMoreOption"><ElIcon :size="20"><MoreFilled /></ElIcon></el-link>
							<el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button> 
							<el-button type="danger" :icon="Download" @click="onDownload">查询下载</el-button> 
						</div> 
				</el-row>
				<el-row :gutter="24" v-if="table_module.queryMoreOption">
					<div class="handle-box">  
						<div   class="formItem">
							<span class="label">自动/人工审核：</span> 
							<el-radio-group   v-model="table_module.query.auditStateEq" text-color="red" class="el-select mr10">
								<el-checkbox v-model="table_module.query.includeAuditStateNull" label="包含空" :disabled="auditStateNullDisabled" />
								<el-radio :label="true"   >相同</el-radio>
								<el-radio :label="false"  >不同</el-radio>
								<el-radio  >不启用</el-radio>
							</el-radio-group>
						</div> 
						<el-select style="width:160px"  class="mr10"  v-model="table_module.query.auditUser" placeholder="审核人员">
							<el-option 
								v-for="item  in form_attach_data.user_name_list" 
								:key="item.key" :label="item.label" :value="item.value"
							/> 
						</el-select> 

						<el-select style="width:160px"  class="mr10"  v-model="table_module.query.breakRules" placeholder="违反规则">
								<el-option 
									v-for="item  in form_attach_data.rule" 
									:key="item.key" :label="item.value" :value="item.key"
								/> 
						</el-select>  
						
					</div>
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
								title="设备时间"
							/>
						</div> 
					</div>
				</el-row>
				<el-row :gutter="24"> 
					 
						<el-table highlight-current-row @sort-change="onColChange"  
						:row-class-name="tableRowProp"
						:data="table_module.tableData" border class="table" ref="tableInstance" @row-click="onTableSelect" header-cell-class-name="table-header">  
							<el-table-column prop="id"  label="ID" width="65" align="center" sortable ></el-table-column>
							<el-table-column prop="boatName" label="船名"  width="90" sortable  :show-overflow-tooltip="true"></el-table-column>
							<el-table-column prop="vioName" label="违规名称" width="110"  sortable :show-overflow-tooltip="true"></el-table-column>
							<el-table-column label="处理状态" width="110"  sortable prop="flowStatus">
								<template #default="scope"> <el-tag  >{{ form_attach_data.getFlowStateName(scope.row.flowStatus)?.label }} </el-tag></template>
							</el-table-column> 
							<el-table-column label="人工审核" width="120" sortable :show-overflow-tooltip="true"  prop="manualAuditResult"> 
								<template #default="scope"> <el-tag :type="form_attach_data.statue2TagType(scope.row.manualAuditResult)">{{ form_attach_data.getManualStateName(scope.row.manualAuditResult)?.label }} </el-tag></template>
							</el-table-column>
							<el-table-column label="程序审核"  width="120" sortable prop="programAuditResult" :show-overflow-tooltip="true">
								<template #default="scope"><el-tag :type="form_attach_data.statue2TagType(scope.row.programAuditResult)"> {{  form_attach_data.getAutoStateName( scope.row.programAuditResult)?.label }} </el-tag></template>
							</el-table-column> 
							<el-table-column width="160" prop="devRecordTime" label="设备时间" sortable  :show-overflow-tooltip="true"></el-table-column> 
							<el-table-column label="操作" width="316" align="center">
							<template #default="scope">  
								<el-button text :icon="Edit" @click="onOpenDialog(  scope.row)" v-permiss="15">
									打标签
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
		<el-dialog title="标注" v-model="form.dialogVisible"   style="width:98%; height: 90%;" @keydown.ctrl="keyDown">
			 <label-process ref="label_view" :options="table_module.currentItem" :meta-data="form_attach_data" @refesh="onRefesh" ></label-process>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="form.dialogVisible = false">取 消</el-button> 
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
import { queryList_svc ,position_svc,start_download_task } from '../api/process';

import  { labelProcess} from '../components/labelprocess';   
import  { types} from '../components/process';   
import { form_attach_data as attach_data  } from "../store/process/viewdata"; 
 
let form_attach_data=reactive<ItemAattachData>( attach_data);
const elTreeInstance=ref<any>(null) 
const selectedVal = ref<any[]>();
const cacheData = [ { value: 5, label: '位置信息' }]
 
const tableInstance=ref<any>(null); 
const currentTableItemIndex = ref<number>();  
const table_module = reactive<table_module>({
	isResearch:true,
	query:{
		boatName: '',
		flowStatus:'', 
		manualAuditStatus:"",
		datetimes:[],
		includeAuditStateNull:false,
		auditStateEq:undefined, 
		breakRules:'',
		pageIndex: 1,
		pageSize: 15,
		order:'asc',
		orderBy:'',
		groupIds:[],
		boatSerials:[], 
		ipCameraSerial:[], 
	},
	queryMoreOption:false,
	cache:{} ,
	currentItem:ref<ProcessTableItem>(), 
	tableData:ref<ProcessTableItem[]>([]),
	pageTotal:-1, 
	//树形选择
	treeData:ref<types.TreeItem[]>([]),
	treeCacheData: [{ value: 5, label: '位置信息' }],  
	treeDataMap:{
		value: 'id',
		label: 'name',
		children: 'children', 
		isLeaf: "isLeaf",
	}, 
});
const auditStateNullDisabled=computed( ()=>{
	return typeof(table_module.query.auditStateEq)!="boolean"
} )
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
	table_module.isResearch=false; 
	getData( );
}
 
// 查询操作
const onSearch = () => {
	table_module.isResearch=true; 
	getData();
}; 
watch(()=> table_module.tableData ,async (newValue?:any, oldValue?:any) => {  
	if(newValue){
		let index=table_module.isResearch?0:currentTableItemIndex.value?.valueOf(); 
		if (index==undefined)index=0;
		await nextTick()   
		setTableSelectItem(index)
	} 
})
const _setQueryContition=()=>{
	table_module.query.pageIndex = table_module.query.pageIndex||1; 
	if(elTreeInstance.value)  { 
		let allCheck:Array<{ groupType: string; serial:string,id:number}>= elTreeInstance._value.getCheckedNodes() 
		table_module.query.groupIds=allCheck.filter(m=>m.groupType=="group"||m.groupType=="company").map(m=>m.id) 
		table_module.query.ipCameraSerial=allCheck.filter(m=>m.groupType=="ch").map(m=>m.serial) 
		table_module.query.boatSerials=allCheck.filter(m=>m.groupType=="site").map(m=>m.serial)   
	}
}
// 获取表格数据
const getData = ( ) => {   
	_setQueryContition(),queryList_svc(table_module.query).then(res => {
        if (res.code==0){
            table_module.tableData = res.data; 
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
	table_module.currentItem=row;
}
 
//tree 
const onLoadTree=(node:TreeNodeData, resolve:any) => { 
  if (node.isLeaf) return resolve([])    
  position_svc({pid:node.data.id}).then( res=>{
    if (res.code==0){
		res.data.forEach((e: { groupType: string; isLeaf:boolean})=> {
			//console.info(e.groupType)
			if(e.groupType=="ch")e.isLeaf=true
			else e.isLeaf=false
		}); 
		resolve(res.data)
	}
  }) 
}  
  
const Oncheck=(data:TreeNodeData,node:{checkedNodes:Array<{id:number,name:string}>} )=>{  
	selectedVal.value=node.checkedNodes.map(m=>m.name)  
}
const process_view=ref()
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
	process_view.value.keyDown(e) 
    e.stopPropagation() 
}
 
const onDownload=()=>{
	ElMessageBox.confirm(`确定以当前条件的创建下载任务吗？`, '提示', {
		type: 'warning'
	})
    .then(() => {
		_setQueryContition(),start_download_task(table_module.query).then(res => {
        if (res.code==0){ 
            ElMessageBox.alert(res.data,`"任务信息"${res.message}`)
        }else{
			ElMessageBox.alert(res.message,"任务创建失败")
        } 
	});
    }) .catch(() => {});  
}
 
//**打标签 */
let dialogData={
	dialogVisible:false, 
}
let form = reactive(dialogData); 
const onOpenDialog=( row?:any)=>{
	form.dialogVisible = true  
	table_module.currentItem=row 
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
