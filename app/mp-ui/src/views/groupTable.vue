<template>
	<div>
		<div class="container">
			<div class="handle-box"> 
				<el-input v-model="query.name" placeholder="名称" class="handle-input mr10"></el-input>
				<el-input v-model="query.boatPosNumber" placeholder="部位编号" class="handle-input mr10"></el-input>
				<el-select style="width:160px"  class="mr10"  v-model="query.type" placeholder="请选择">
					<el-option 
						v-for="item in from_attach_data.group" 
						:key="item.key" :label="item.label" :value="item.key"
					/> 
				</el-select>  
				<el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button> 
			</div>
			<el-table 
				:data="tableData"  @sort-change="onColChange"
				border class="table" ref="multipleTable" header-cell-class-name="table-header"> 
				<el-table-column label="序号" width="55" align="center">
					<template #default="scope"> {{scope.$index}} </template>
				</el-table-column> 
				<el-table-column prop="name" label="名称" sortable :show-overflow-tooltip="true"></el-table-column>
				<el-table-column label="分组类型" sortable :show-overflow-tooltip="true">
					<template #default="scope">{{scope.row.groupType  }} {{ from_attach_data.getGroupItem(scope.row.groupType)?.label }} </template>
				</el-table-column>  
				<el-table-column label="序列号" sortable :show-overflow-tooltip="true">
					<template #default="scope"> {{  scope.row.ipCameraSerial|| scope.row.boatSerial  }} </template>
				</el-table-column>   
				<el-table-column label="部位编号" sortable :show-overflow-tooltip="true" prop="boatPosNumber"></el-table-column> 
				<el-table-column prop="updateTime" label="更新时间" sortable :show-overflow-tooltip="true"></el-table-column> 
				<el-table-column label="操作" width="316" align="center">
					<template #default="scope">  
						<el-button text :icon="Setting" v-if="from_attach_data.allowSetNumberGroup.indexOf( scope.row.groupType)>-1" @click="onOpenEditDialog(scope.$index, scope.row)"  >
							设置编号
						</el-button> 
					</template>
				</el-table-column>
			</el-table>
			<div class="pagination">
				<el-pagination
					background
					layout="total, prev, pager, next"
					:current-page="query.pageIndex"
					:page-size="query.pageSize"
					:total="pageTotal"
					@current-change="onCurrentPageChange"
				></el-pagination>
			</div>
		</div> 

		<!-- 编辑弹出框 -->
	<el-dialog title="编辑" v-model="editVisible" width="30%">
			<el-form label-width="70px">
				<el-form-item label="船名">
					<el-input readonly v-model="form.boartName"></el-input>
				</el-form-item>
				<el-form-item label="船序列号">
					<el-input readonly v-model="form.boartSerial"></el-input>
				</el-form-item>
				
				<el-form-item label="设备名">
					<el-input readonly v-model="form.name"></el-input>
				</el-form-item>
				<el-form-item label="序列号">
					<el-input readonly v-model="form.serial"></el-input>
				</el-form-item>
				<el-form-item label="部位编号">
					<el-input v-model="form.boatPosNumber" placeholder="zhc_jh36_jb[船类型_船编号_部位名称]"></el-input>
				</el-form-item>  
			</el-form>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="editVisible = false">取 消</el-button>
					<el-button type="primary" @click="onEditSave">确 定</el-button>
				</span>
			</template>
		</el-dialog>
	</div>

	
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue';
import { ElMessage, ElMessageBox,FormRules,FormInstance } from 'element-plus';
import { Delete, Edit, Search, Compass ,Plus,Setting} from '@element-plus/icons-vue'; 
import { queryList_svc,update_svc ,get_one_svc} from '../api/group'; 
import { attach_data,AtachData } from "../store/boatgroup"; 

let from_attach_data=reactive<AtachData>( attach_data);

interface QueryType{
	name?:string,
	type?:string,
	boatPosNumber?:string,
	pageIndex:number,
	pageSize:number 
	order:'asc'|'desc',
	orderBy:string,
}
const query = reactive<QueryType>({  
	pageIndex: 1,
	pageSize: 10,
	order:'asc',
	orderBy:'',
});
const tableData = ref<any[]>([]);
const pageTotal = ref(0);
// 获取表格数据
const getData = () => {
	queryList_svc(query).then(res => {
        if (res.code==0){
            tableData.value = res.data;
		    pageTotal.value = res.total || -1;
        }else{
            ElMessage.error(res.message); 
        } 
	});
};
getData();
const onColChange = (column:any) =>  { 
	//console.info(column) 
	query.order = column.order  === 'descending' ? 'desc' : 'asc'
	query.orderBy = column.prop 
	if (column) getData() // 获取数据的方法
} 
// 查询操作
const onSearch = () => {
	query.pageIndex = 1;
	getData();
};
// 分页导航
const onCurrentPageChange = (val: number) => {
	query.pageIndex = val;
	getData();
};
const editVisible = ref(false); 
let fromData={
    id:-1,
	boartName:'',
	name: '',
	serial  :'',
    boatPosNumber:"",  
	boartSerial:""
}
let form = reactive(fromData);
const onOpenEditDialog =async (index: number, row: any) => { 
    form.id=row.id
	form.name = row.name;
	form.serial = row.ipCameraSerial;
    form.boatPosNumber = row.boatPosNumber;
	let res=await get_one_svc(row.pId) 
	form.boartName =res.data.name;
	form.boartSerial =res.data.boatSerial;
	editVisible.value = true;
};
const onEditSave = () => { 
    update_svc(form.id,{boatPosNumber:form.boatPosNumber }).then(res=>{
        if(res.code==0){
			editVisible.value = false; 
            ElMessage.success(`修改‘${form.name}’成功`);
            getData()
        }
        else {ElMessage.error(`修改‘${form.name}’修改失败:${res.message}`);}
    }) 
};
</script>

<style scoped>
.handle-box {
	margin-bottom: 20px;
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
</style>
