<template>
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						v-model="table_module.query.name"
						placeholder="菜单名称"
						class="handle-input"></el-input>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>搜索</el-button
					>
					<el-button
						type="primary"
						:icon="Plus"
						@click="onOpenDialog(0)"
						v-permiss="getPermissKey(routeHook.ViewFeature.add)"
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
							<template #default="scope">
								{{ getTableIndex(table_module.query, scope.$index) }}
							</template>
						</el-table-column>

						<el-table-column
							prop="name"
							label="菜单名称"
							sortable
							:show-overflow-tooltip="true" />
						<el-table-column
							label="公众号"
							width="110"
							sortable
							prop="flowStatus">
							<template #default="scope">
								<el-tag
									>{{ store.getItem(scope.row.openId)?.name }}
								</el-tag></template
							>
						</el-table-column>
						<el-table-column
							prop="state"
							label="状态"
							sortable
							show-overflow-tooltip
							:show-after="2000">
							<template #default="scope">
								<el-tag>
									{{ store.getMenuStateItem(scope.row.state)?.label }}
								</el-tag>
							</template>
						</el-table-column>

						<el-table-column
							prop="createTime"
							label="创建时间"
							sortable
							:show-overflow-tooltip="true" />
						<el-table-column
							prop="updateTime"
							label="更新时间"
							sortable
							:show-overflow-tooltip="true" />
						<el-table-column
							label="操作"
							width="400"
							fixed="right"
							align="left">
							<template #default="scope">
								<el-button
									text
									:icon="Edit"
									v-permiss="getPermissKey(routeHook.ViewFeature.edit)"
									@click="onOpenDialog(1, scope.row)">
									编辑
								</el-button>
								<el-button
									text
									:icon="Compass"
									v-if="scope.row.openId"
									v-permiss="getPermissKey(routeHook.ViewFeature.push)"
									@click="onPush(scope.$index, scope.row)">
									推送
								</el-button>
								<el-button
									text
									:icon="Plus"
									title="获取当前公众号配置的菜单"
									v-if="scope.row.openId"
									v-permiss="getPermissKey(routeHook.ViewFeature.get)"
									@click="onPull(scope.$index, scope.row)">
									获取
								</el-button>
								<el-button
									text
									:icon="Plus"
									title="重置公众号菜单"
									v-if="scope.row.openId"
									v-permiss="getPermissKey(routeHook.ViewFeature.reset)"
									@click="onReset(scope.$index, scope.row)">
									重置
								</el-button>
								<el-button
									text
									:icon="Delete"
									class="red"
									v-permiss="getPermissKey(routeHook.ViewFeature.del)"
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
						layout="total, prev, pager, next"
						:current-page="table_module.query.pageIndex"
						:page-size="table_module.query.pageSize"
						:total="table_module.pageTotal"
						@current-change="onPageChange"></el-pagination>
				</div>
			</el-footer>
		</el-container>
		<!-- 弹出框 -->
		<modifyDiaglog
			style="width: 80%; height: 80%"
			ref="modifyDiaglogRef"
			@saved="onSearch"
			title="编辑" />
	</div>
</template>

<script setup lang="ts" name="basetable">
	import { ref, reactive, onMounted } from 'vue';

	import {
		ElMessage,
		ElInput,
		ElButton,
		ElHeader,
		ElTableColumn,
		ElTag,
		ElTable,
		ElScrollbar,
		ElMain,
		ElPagination,
		ElFooter,
		ElContainer,
	} from 'element-plus';
	import { Delete, Edit, Search, Compass, Plus } from '@element-plus/icons-vue';
	import { menu as mp_api } from '@/api';
	import { get_store } from '@/hooks/wx';
	import { warningArgs, EleConfirm, getTableIndex } from 'co6co';
	import { routeHook } from 'co6co-right';

	import modifyDiaglog, {
		type MenuItem as Item,
		type ModifyMenuInstance,
	} from '@/components/modifyMenu';

	import {
		showLoading,
		closeLoading,
		FormOperation,
		type IPageParam,
		type Table_Module_Base,
	} from 'co6co';
	const { getPermissKey } = routeHook.usePermission();
	const store = get_store();

	interface IQueryItem extends IPageParam {
		name?: string;
	}

	interface Table_Module extends Table_Module_Base {
		query: IQueryItem;
		data: Item[];
		currentItem?: Item;
	}

	const table_module = reactive<Table_Module>({
		query: {
			name: '',
			pageIndex: 1,
			pageSize: 15,
			order: 'asc',
			orderBy: '',
		},
		data: [],
		pageTotal: -1,
		diaglogTitle: '',
	});
	// 获取表格数据
	const getData = async () => {
		showLoading();
		store
			.refesh()
			.then(() => mp_api.list_menu_svc(table_module.query))
			.then((res) => {
				table_module.data = res.data;
				table_module.pageTotal = res.total || -1;
			})
			.finally(() => {
				closeLoading();
			});
	};

	// 查询操作
	const onSearch = () => {
		getData();
	};

	const onColChange = (column: any) => {
		table_module.query.order = column.order === 'descending' ? 'desc' : 'asc';
		table_module.query.orderBy = column.prop;
		if (column) getData(); // 获取数据的方法
	};
	const onPageChange = (val: number) => {
		table_module.query.pageIndex = val;
		getData();
	};

	const modifyDiaglogRef = ref<ModifyMenuInstance>();

	//编辑角色
	const onOpenDialog = (operation: FormOperation, row?: Item) => {
		table_module.diaglogTitle =
			operation == FormOperation.add ? '增加菜单' : `编辑[${row?.name}]菜单`;
		table_module.currentItem = row;
		modifyDiaglogRef.value?.openDialog(operation, row);
	};

	// 删除操作
	const onDelete = (index: number, row: any) => {
		// 二次确认删除
		EleConfirm(`确定要删除"${row.name}"菜单吗？`, { ...warningArgs })
			.then(() => {
				showLoading();
				return mp_api.del_menu_svc(row.id);
			})
			.then(() => {
				closeLoading();
				ElMessage.success('删除成功'), getData();
			});
	};

	//推送菜单
	const onPush = (index: number, row: any) => {
		EleConfirm(`确定要推送"${row.name}"到微信公众号吗？`, { ...warningArgs })
			.then(() => {
				showLoading();
				return mp_api.push_menu_svc(row.id);
			})
			.then(() => {
				closeLoading();
				ElMessage.success('推送成功'), getData();
			});
	};
	const onPull = (index: number, row: any) => {
		EleConfirm(`确定要获取微信公众号菜单？`, { ...warningArgs })
			.then(() => {
				showLoading();
				return mp_api.pull_menu_svc(row.id);
			})
			.then(() => {
				closeLoading();
				ElMessage.success('获取成功'), getData();
			});
	};
	const onReset = (index: number, row: any) => {
		EleConfirm(`确定要重置公众号菜单，重置后菜单降不存在！`, { ...warningArgs })
			.then(() => {
				showLoading();
				return mp_api.reset_menu_svc(row.id);
			})
			.then(() => {
				closeLoading();
				ElMessage.success('重置成功'), getData();
			});
	};
	//弹出框 add and edit
	onMounted(() => {
		getData();
	});
</script>
<style scoped>
	.cell .el-button {
		padding: 8px 7px;
	}
</style>
