<template>
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						v-model="query.name"
						placeholder="菜单名称"
						class="handle-input mr10"></el-input>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>搜索</el-button
					>
					<el-button type="primary" :icon="Plus" @click="onOpenDialog(0)"
						>新增</el-button
					>
				</div>
			</el-header>
			<el-main>
				<el-scrollbar>
					<el-table
						highlight-current-row
						@sort-change="onColChange"
						:data="table_module.data"
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
						<el-table-column
							prop="name"
							label="菜单名称"
							sortable
							:show-overflow-tooltip="true"></el-table-column>
						<el-table-column
							label="公众号"
							width="110"
							sortable
							prop="flowStatus">
							<template #default="scope">
								<el-tag
									>{{ config.getItem(scope.row.openId)?.name }}
								</el-tag></template
							>
						</el-table-column>
						<el-table-column
							prop="state"
							label="状态"
							sortable
							:show-overflow-tooltip="true">
							<template #default="scope">
								<el-tag
									>{{ config.getMenuStateItem(scope.row.state)?.label }}
								</el-tag></template
							>
						</el-table-column>

						<el-table-column
							prop="createTime"
							label="创建时间"
							sortable
							:show-overflow-tooltip="true"></el-table-column>
						<el-table-column
							prop="updateTime"
							label="更新时间"
							sortable
							:show-overflow-tooltip="true"></el-table-column>
						<el-table-column label="操作" width="388" align="left">
							<template #default="scope">
								<el-button
									text
									:icon="Edit"
									@click="onOpenDialog(1, scope.row)">
									编辑
								</el-button>
								<el-button
									text
									:icon="Compass"
									v-if="scope.row.openId"
									@click="onPush(scope.$index, scope.row)">
									推送
								</el-button>
								<el-button
									text
									:icon="Plus"
									title="获取当前公众号配置的菜单"
									v-if="scope.row.openId"
									@click="onPull(scope.$index, scope.row)">
									获取
								</el-button>
								<el-button
									text
									:icon="Delete"
									class="red"
									@click="onDelete(scope.$index, scope.row)">
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
						layout="prev, pager, next,total,jumper"
						:current-page="query.pageIndex"
						:page-size="query.pageSize"
						:total="pageTotal"
						@current-change="onCurrentPageChange"></el-pagination>
				</div>
			</el-footer>
		</el-container>
		<!-- 弹出框 -->
		<el-dialog :title="form.title" v-model="form.dialogVisible" width="80%">
			<el-form
				label-width="90px"
				ref="dialogForm"
				:rules="rules"
				:model="form.fromData">
				<el-form-item label="所属公众号" prop="openId">
					<el-select
						style="width: 160px"
						class="mr10"
						v-model="form.fromData.openId"
						placeholder="请选择">
						<el-option
							v-for="item in config.list"
							:key="item.openId"
							:label="item.name"
							:value="item.openId" />
					</el-select>
				</el-form-item>
				<el-form-item label="名称" prop="name">
					<el-input
						v-model="form.fromData.name"
						placeholder="菜单名称"></el-input>
				</el-form-item>
				<el-form-item label="内容" prop="content">
					<md-editor
						:preview="false"
						class="mgb20"
						v-model="form.fromData.content" />
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
	import { ref, reactive, watchEffect } from 'vue';
	import {
		ElMessage,
		ElMessageBox,
		FormRules,
		FormInstance,
		ElLoading,
	} from 'element-plus';
	import {
		Delete,
		Edit,
		Search,
		Compass,
		Plus,
		Download,
	} from '@element-plus/icons-vue';
	import {
		get_config_svc,
		list_menu_svc,
		add_menu_svc,
		edit_menu_svc,
		del_menu_svc,
		push_menu_svc,
		pull_menu_svc,
	} from '../api/wx';
	import { wx_config_store } from '../store/wx';
	import MdEditor from 'md-editor-v3';
	import 'md-editor-v3/lib/style.css';
	import { showLoading, closeLoading } from '../components/Logining';
	interface TableRow {
		id: number;
		name: string;
		openId: string;
	}

	interface Query extends IpageParam {
		name: '';
	}
	const query = reactive<Query>({
		name: '',
		pageIndex: 1,
		pageSize: 10,
		order: '',
	});

	const config = wx_config_store();
	const pageTotal = ref(0);

	interface table_module {
		query: Query;
		data: TableRow[];
		currentRow?: TableRow;
		pageTotal: number;
	}

	const table_module = reactive<table_module>({
		query: {
			name: '',
			pageIndex: 1,
			pageSize: 15,
			order: 'asc',
			orderBy: '',
		},
		data: [],
		pageTotal: 0,
	});
	// 获取表格数据
	const getData = async () => {
		showLoading();
		await config.refesh();
		list_menu_svc(query)
			.then((res) => {
				if (res.code == 0) {
					table_module.data = res.data;
					table_module.pageTotal = res.total || -1;
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
	const onColChange = (column: any) => {
		//console.info(column)
		query.order = column.order === 'descending' ? 'desc' : 'asc';
		query.orderBy = column.prop;
		if (column) getData(); // 获取数据的方法
	};
	// 分页导航
	const onCurrentPageChange = (val: number) => {
		query.pageIndex = val;
		getData();
	};

	// 删除操作
	const onDelete = (index: number, row: any) => {
		// 二次确认删除
		ElMessageBox.confirm(`确定要删除"${row.name}"任务吗？`, '提示', {
			type: 'warning',
		})
			.then(() => {
				del_menu_svc(row.id)
					.then((res) => {
						if (res.code == 0) ElMessage.success('删除成功'), getData();
						else ElMessage.error(`删除失败:${res.message}`);
					})
					.finally(() => {});
			})
			.catch(() => {});
	};

	//推送菜单
	const onPush = (index: number, row: any) => {
		//
		ElMessageBox.confirm(`确定要推送"${row.name}"到微信公众号吗？`, '提示', {
			type: 'warning',
		})
			.then(() => {
				push_menu_svc(row.id)
					.then((res) => {
						if (res.code == 0) ElMessage.success('推送成功'), getData();
						else ElMessage.error(`推送失败:${res.message}`);
					})
					.finally(() => {});
			})
			.catch(() => {});
	};
	const onPull = (index: number, row: any) => {
		//
		ElMessageBox.confirm(`确定要获取微信公众号菜单？`, '提示', {
			type: 'warning',
		})
			.then(() => {
				pull_menu_svc(row.id)
					.then((res) => {
						if (res.code == 0) ElMessage.success('获取成功'), getData();
						else ElMessage.error(`获取失败:${res.message}`);
					})
					.finally(() => {});
			})
			.catch(() => {});
	};
	//弹出框 add and edit
	const dialogForm = ref<FormInstance>();
	const rules: FormRules = {
		name: [{ required: true, message: '请输入名称', trigger: ['blur'] }],
		alias: [{ required: true, message: '请输入别名', trigger: 'blur' }],
	};
	let dialogData = {
		dialogVisible: false,
		title: '',
		operation: 0,
		id: -1,
		fromData: {
			openId: '',
			name: '',
			content: '',
		},
	};
	let form = reactive(dialogData);
	const onOpenDialog = (type: number, row?: any) => {
		form.dialogVisible = true;
		form.operation = type;
		form.id = -1;
		switch (type) {
			case 0:
				form.title = '增加';
				form.fromData.openId = '';
				form.fromData.name = '';
				form.fromData.content = '';
				break;
			case 1:
				form.id = row.id;
				form.title = '编辑';
				form.fromData.openId = row.openId;
				form.fromData.name = row.name;
				form.fromData.content = row.content;
				break;
		}
	};
	const onDialogSave = (formEl: FormInstance | undefined) => {
		if (!formEl) return;
		formEl.validate((value) => {
			if (value) {
				if (form.operation == 0) {
					add_menu_svc(form.fromData).then((res) => {
						if (res.code == 0) {
							form.dialogVisible = false;
							ElMessage.success(`增加成功`);
							getData();
						} else {
							ElMessage.error(`增加失败:${res.message}`);
						}
					});
				} else {
					edit_menu_svc(form.id, form.fromData).then((res) => {
						if (res.code == 0) {
							form.dialogVisible = false;
							ElMessage.success(`编辑成功`);
							getData();
						} else {
							ElMessage.error(`编辑失败:${res.message}`);
						}
					});
				}
			} else {
				ElMessage.error('请检查输入的数据！');
				return false;
			}
		});
	};
</script>

<style scoped lang="less">
	@import '../assets/css/tables.css';
</style>
