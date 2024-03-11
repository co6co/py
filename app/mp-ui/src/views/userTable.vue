<template>
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						v-model="query.name"
						placeholder="用户名称"
						class="handle-input mr10"></el-input>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>搜索</el-button
					>
					<el-button type="primary" :icon="Plus" @click="onOpenDialog()"
						>新增</el-button
					>
				</div>
			</el-header>
			<el-main>
				<el-scrollbar>
					<el-table
						:data="tableData"
						border
						class="table"
						ref="multipleTable"
						header-cell-class-name="table-header">
						<el-table-column label="序号" width="55" align="center">
							<template #default="scope"> {{ scope.$index }} </template>
						</el-table-column>
						<el-table-column
							prop="id"
							label="ID"
							width="55"
							align="center"></el-table-column>
						<el-table-column prop="userName" label="用户名"></el-table-column>
						<el-table-column label="用户组" prop="groupName"> </el-table-column>
						<el-table-column label="微信昵称" prop="nickName"> </el-table-column> 
						<el-table-column
							label="所属公众号"
							width="120"
							sortable
							prop="flowStatus">
							<template #default="scope">
								<el-tag
									>{{ config.getItem(scope.row.ownedAppid)?.name }}
								</el-tag></template
							>
							</el-table-column>
						<el-table-column label="状态" align="center">
							<template #default="scope">
								<el-tag>
									{{m.state.getStateName(scope.row.state)?.label }}
								</el-tag>
							</template>
						</el-table-column>

						<el-table-column
							prop="createTime"
							label="注册时间"></el-table-column>
						<el-table-column label="操作" width="316" align="center">
							<template #default="scope">
								<el-button
									text
									:icon="Edit"
									@click="onOpenDialog(  scope.row)"
									v-permiss="15">
									编辑
								</el-button>
								<el-button
									text
									:icon="Delete"
									class="red"
									@click="onDelete(scope.$index, scope.row)"
									v-permiss="16">
									删除
								</el-button>
								<el-button
									text
									:icon="Compass"
									@click="onOpenResetDialog(scope.$index, scope.row)"
									v-permiss="15">
									重置密码
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
						layout="prev, pager, next,total,jumper"
						:current-page="query.pageIndex"
						:page-size="query.pageSize"
						:total="pageTotal"
						@current-change="handlePageChange"></el-pagination>
				</div>
			</el-footer>
		</el-container>

		 

		<edit-user ref="editUserRef"  @saved="getData()"></edit-user>

		<el-dialog title="重置密码" v-model="resetPwdVisible" width="30%">
			<el-form label-width="70px" :model="resetPwdFrom" :rules="resetPwdRules">
				<el-form-item label="用户名" prop="userName">
					<el-input readonly v-model="resetPwdFrom.userName"></el-input>
				</el-form-item>
				<el-form-item label="新密码" prop="password">
					<el-input
						v-model="resetPwdFrom.password"
						type="password"
						show-password>
					</el-input>
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
	import {
		ElMessage,
		ElMessageBox,
		type FormRules,
		type FormInstance,
	} from 'element-plus';
	import { Delete, Edit, Search, Compass, Plus } from '@element-plus/icons-vue';
	import {
		queryList_svc,
		exist_svc,
		add_svc,
		edit_svc,
		del_svc,
		retsetPwd_svc,
	} from '../api/user';

	import { showLoading, closeLoading } from '../components/Logining';
	import { editUser,model as m } from '../components/users';
	import { wx_config_store } from '../store/wx';

	interface TableItem {
		id: number;
		userName: string;
		state: number;
		roleId: number;
	}

	const query = reactive({
		name: '',
		pageIndex: 1,
		pageSize: 10,
	});
	const config = wx_config_store();
	const tableData = ref<TableItem[]>([]);
	const pageTotal = ref(0);
	// 获取表格数据
	const getData = async () => {
		showLoading();
		await config.getConfig();
		queryList_svc(query)
			.then((res) => {
				if (res.code == 0) {
					tableData.value = res.data;
					pageTotal.value = res.total || -1;
				} else {
					ElMessage.error(res.message);
				}
			})
			.finally(() => {
				closeLoading();
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
			type: 'warning',
		})
			.then(() => {
				del_svc(row.id)
					.then((res) => {
						if (res.code == 0) ElMessage.success('删除成功'), getData();
						else ElMessage.error(`删除失败:${res.message}`);
					})
					.finally(() => {});
			})
			.catch(() => {});
	};



	//编辑增加用户
	const editUserRef=ref()
	const onOpenDialog = (row?: any) => { 
		//有记录编辑无数据增加  
		console.info("ddd",row)
		editUserRef.value.onOpenDialog(row?1:0,row);
	};  

	//重置密码
	let resetPwdVisible = ref(false);
	let resetPwdFrom = reactive({
		userName: '',
		password: '',
		oldPassword: '',
	});
	const resetPwdRules: FormRules = {
		userName: [{ required: true, message: '请输入用户名', trigger: ['blur'] }],
		password: [
			{
				required: true,
				min: 6,
				max: 20,
				message: '请输入6-20位新密码',
				trigger: ['blur', 'change'],
			},
		],
	};
	const onOpenResetDialog = (index: number, row: any) => {
		resetPwdVisible.value = true;
		resetPwdFrom.userName = row.userName;
		resetPwdFrom.password = '';
	};
	const onResetPwdSave = () => {
		retsetPwd_svc(resetPwdFrom).then((res) => {
			if (res.code == 0) {
				resetPwdVisible.value = false;
				ElMessage.success(`重置密码成功`);
			} else {
				ElMessage.error(`重置密码失败:${res.message}`);
			}
		});
	}; 
</script>

<style scoped lang="less">
	@import '../assets/css/tables.css';
</style>
