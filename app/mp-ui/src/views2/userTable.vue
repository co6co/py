<template>
	<div>
		<div class="container">
			<div class="handle-box"> 
				<el-input v-model="query.name" placeholder="用户名" class="handle-input mr10"></el-input>
				<el-button type="primary" :icon="Search" @click="onSearch">搜索</el-button>
				<el-button type="primary" :icon="Plus" @click="onOpenAddDialog">新增</el-button>
			</div>
			<el-table :data="tableData" border class="table" ref="multipleTable" header-cell-class-name="table-header"> 
				<el-table-column   label="序号" width="55" align="center">
					<template #default="scope"> {{scope.$index}} </template>
				</el-table-column>
				<el-table-column prop="id" label="ID" width="55" align="center"></el-table-column>
				<el-table-column prop="userName" label="用户名"></el-table-column>
				<el-table-column label="用户组" prop="groupName"> </el-table-column>  
				<el-table-column label="状态" align="center">
					<template #default="scope">
						<el-tag >
                        {{ form_attach_data.getStateName(scope.row.state)?.label }} 
						</el-tag>
					</template>
				</el-table-column>

				<el-table-column prop="createTime" label="注册时间"></el-table-column>
				<el-table-column label="操作" width="316" align="center">
					<template #default="scope"> 
						<el-button text :icon="Edit" @click="onOpenEditDialog(scope.$index, scope.row)" v-permiss="15">
							编辑
						</el-button>
						<el-button text :icon="Delete" class="red" @click="onDelete(scope.$index,scope.row)" v-permiss="16">
							删除
						</el-button>
						<el-button text :icon="Compass" @click="onOpenResetDialog(scope.$index, scope.row)" v-permiss="15">
							重置密码
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
					@current-change="handlePageChange"
				></el-pagination>
			</div>
		</div>


		<!-- 增加弹出框 -->
		<el-dialog title="增加" v-model="addVisible" width="30%">
			<el-form label-width="80px" :model="form" :rules="rules" ref="addForm">
				<el-form-item label="用户名" prop="userName">
					<el-input v-model="form.userName"></el-input>
				</el-form-item>
				<el-form-item label="密码" prop="password">
					<el-input  v-model="form.password" type="password" show-password>  </el-input>
				</el-form-item>
				<el-form-item label="用户角色" prop="roleId"> 
                    <el-select v-model="form.roleId" placeholder="请选择">
                        <el-option v-for="item  in form_attach_data.roleList" 
                            :key="item.key" :label="item.label" :value="item.value"
                        /> 
                    </el-select> 
				</el-form-item>
                <el-form-item label="用户状态" prop="state">
                    <el-select v-model="form.state" placeholder="请选择">
                        <el-option v-for="item  in form_attach_data.stateList" 
                            :key="item.key" :label="item.label" :value="item.value"
                        />  
                    </el-select> 
				</el-form-item>
			</el-form>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="addVisible = false">取 消</el-button>
					<el-button type="primary" @click="onAddSave(addForm)">确 定</el-button>
				</span>
			</template>
		</el-dialog>

		<!-- 编辑弹出框 -->
		<el-dialog title="编辑" v-model="editVisible" width="30%">
			<el-form label-width="70px">
				<el-form-item label="用户名">
					<el-input readonly v-model="form.userName"></el-input>
				</el-form-item>
				<el-form-item label="用户角色"> 
                    <el-select v-model="form.roleId" placeholder="请选择"> 
                        <el-option v-for="item  in form_attach_data.roleList" 
                            :key="item.key" :label="item.label" :value="item.value" 
                        /> 
                    </el-select> 
				</el-form-item>
                <el-form-item label="用户状态">
                    <el-select v-model="form.state" placeholder="请选择">
                        <el-option v-for="item  in form_attach_data.stateList" 
                            :key="item.key" :label="item.label" :value="item.value"
                        />  
                    </el-select> 
				</el-form-item>
			</el-form>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="editVisible = false">取 消</el-button>
					<el-button type="primary" @click="onEditSave">确 定</el-button>
				</span>
			</template>
		</el-dialog>

		<el-dialog title="重置密码"  v-model="resetPwdVisible" width="30%">
			<el-form label-width="70px" :model="resetPwdFrom" :rules="resetPwdRules">
				<el-form-item label="用户名" prop="userName">
					<el-input readonly v-model="resetPwdFrom.userName"></el-input>
				</el-form-item> 
				<el-form-item label="新密码" prop="password">
					<el-input  v-model="resetPwdFrom.password" type="password" show-password>  </el-input>
				</el-form-item>
			</el-form>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="resetPwdVisible = false">取 消</el-button>
					<el-button type="primary" @click="onResetPwdSave">确 定</el-button>
				</span>
			</template>
		</el-dialog>
	</div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue';
import { ElMessage, ElMessageBox, type FormRules,type FormInstance } from 'element-plus';
import { Delete, Edit, Search, Compass ,Plus} from '@element-plus/icons-vue'; 
import { queryList_svc, exist_svc,add_svc, edit_svc,del_svc ,retsetPwd_svc} from '../api/user'; 

interface TableItem {
	id: number;
    userName: string; 
	state: number; 
    roleId :number
}

const query = reactive({ 
	name: '',
	pageIndex: 1,
	pageSize: 10
});
const tableData = ref<TableItem[]>([]);
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

// 查询操作
const onSearch = () => {
	query.pageIndex = 1;
	getData();
};
// 分页导航
const handlePageChange = (val: number) => {
	query.pageIndex = val;
	getData();
};

// 删除操作
const onDelete = (index: number, row: any) => {
	// 二次确认删除
	ElMessageBox.confirm(`确定要删除"${row.userName}"吗？`, '提示', {
		type: 'warning'
	})
    .then(() => {
        del_svc(row.id).then(res=>{
            if (res.code==0)  ElMessage.success('删除成功'),getData(); 
            else  ElMessage.error(`删除失败:${res.message}`); 
        }) .finally(()=>{

		})
    })
    .catch(() => {});
};
 
let fromData={
    id:-1,
	userName: '',
	password:'',
    roleId:-1, 
	state: -1
}
let form = reactive(fromData);
let form_attach_data=reactive({
	roleList:   [
        { key:"超级用户",label:"超级用户",value:1},
        { key:"操作员",label:"操作员",value:2}
    ] ,
    getRoleName(v:number){ 
		return this.roleList.find(m=>m.value==v)
	},
    stateList:[ 
        { key:"启用",label:"启用",value:0},
        { key:"锁定",label:"锁定",value:1},
        { key:"禁用",label:"禁用",value:2} 
    ],
    getStateName(v:number){ 
		return this.stateList.find(m=>m.value==v)
	},
})
//add
const valid_userName=(rule:any,value:string,back:Function)=>{ 
	if(value.length<3) return rule.message="长度应该大于3", back(new Error(rule.message))
	exist_svc(value).then(res=>{
		if(res.code==0) return rule.message=res.message, back(new Error(rule.message))
		else return back( );
	}) 
}
const addForm = ref<FormInstance>();
const addVisible = ref(false);
const rules: FormRules ={
	userName: [ { required: true, validator:valid_userName,  message: '请输入用户名', trigger: ['blur'] } ],
	password: [{ required: true, min:6 ,max:20, message: '请输入6-20位密码', trigger:  'blur' }], 
	roleId:  [{ required: true, message: '请选择用户角色', trigger:  ['blur','change'] }],
	state:    [{ required: true, message: '请选择用户状态', trigger:  ['blur','change'] }], 
};
const onOpenAddDialog=()=>{
	addVisible.value = true  
	form.userName = "";
	form.password=""; 
    form.roleId = 2;
	form.state = 0; 
}
const onAddSave = (formEl: FormInstance | undefined) => { 
	if( !formEl)return
	formEl.validate(value=>{
		if(value){
			add_svc(form).then(res=>{
				if(res.code==0){
					addVisible.value = false; 
					ElMessage.success(`增加用户成功`);
					getData()
				}
				else {ElMessage.error(`增加用户失败:${res.message}`);}
			}) 
		}else{
			ElMessage.error("请检查输入的数据！")
			return false;
		} 
	})
};
//重置密码
let resetPwdVisible=ref(false);
let resetPwdFrom = reactive({
	userName:"",
	password:"",
	oldPassword:""
});
const resetPwdRules: FormRules ={
	userName: [ { required: true,  message: '请输入用户名', trigger: ['blur'] } ],
	password: [{ required: true, min:6 ,max:20, message: '请输入6-20位新密码', trigger:  ['blur',"change"] }] 
};
const onOpenResetDialog=(index: number, row: any)=>{
	resetPwdVisible.value=true;
	resetPwdFrom.userName=row.userName;
	resetPwdFrom.password=""; 
}
const onResetPwdSave=( )=>{
	retsetPwd_svc(resetPwdFrom).then(res=>{
		if(res.code==0){
			resetPwdVisible.value = false; 
            ElMessage.success(`重置密码成功`); 
        }
        else {ElMessage.error(`重置密码失败:${res.message}`);}
	}) 
}
// 表格编辑时弹窗和保存
const editVisible = ref(false);

let idx: number = -1;
const onOpenEditDialog = (index: number, row: any) => {
	idx = index;
    form.id=row.id
	form.userName = row.userName;
	form.state = row.state;
    form.roleId = row.roleId;
	editVisible.value = true;
};
const onEditSave = () => { 
    edit_svc(form.id,{userName:form.userName,roleId:form.roleId,state:form.state}).then(res=>{
        if(res.code==0){
			editVisible.value = false; 
            ElMessage.success(`修改‘${form.userName}’第 ${idx + 1}行成功,修改了功能相关的,需要注销，重新登录才会生效`);
            getData()
        }
        else {ElMessage.error(`修改‘${form.userName}’第 ${idx + 1} 行修改失败:${res.message}`);}
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
