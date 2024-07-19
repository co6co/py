<template>
	<div class="container-layout">
		<ElContainer>
			<el-header>
				<div class="handle-box">
					<el-input
						style="width: 160px"
						v-model="table_module.query.name"
						clearable
						placeholder="菜单名称"
						class="handle-input mr10"></el-input>
					<el-input
						style="width: 160px"
						clearable
						v-model="table_module.query.code"
						placeholder="菜单编码"
						class="handle-input mr10"></el-input>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>查询</el-button
					>
					<el-button
						v-permiss="getPermissKey(ViewFeature.add)"
						type="primary"
						:icon="Plus"
						@click="onOpenDialog(FormOperation.add)"
						>新增</el-button
					>
				</div>
			</el-header>
			<ElContainer>
				<ElContainer>
					<el-main>
						<el-scrollbar>
							<el-table
								:data="table_module.data"
								@sort-change="onColChange"
								border
								class="table"
								header-cell-class-name="table-header"
								row-key="id"
								:tree-props="{ children: 'children' }">
								<el-table-column label="序号" width="100" align="center">
									<template #default="scope"> {{ scope.$index + 1 }} </template>
								</el-table-column>
								<el-table-column
									prop="name"
									label="名称"
									sortable="custom"
									:show-overflow-tooltip="true"></el-table-column>
								<el-table-column
									width="110"
									label="父节点"
									prop="parentId"
									sortable="custom"
									:show-overflow-tooltip="true">
									<template #default="scope">
										{{ getName(scope.row.parentId) || '-' }}
									</template>
								</el-table-column>
								<el-table-column
									prop="code"
									label="代码"
									sortable="custom"
									:show-overflow-tooltip="true"></el-table-column>
								<el-table-column
									prop="order"
									label="排序"
									sortable="custom"
									:show-overflow-tooltip="true"></el-table-column>

								<el-table-column
									width="156"
									prop="createTime"
									label="创建时间"
									sortable="custom"
									:show-overflow-tooltip="true"></el-table-column>
								<el-table-column
									width="156"
									prop="updateTime"
									label="更新时间"
									sortable="custom"
									:show-overflow-tooltip="true"></el-table-column>
								<el-table-column
									label="操作"
									width="216"
									align="center"
									fixed="right">
									<template #default="scope">
										<el-button
											v-permiss="getPermissKey(ViewFeature.edit)"
											text
											:icon="Setting"
											@click="onOpenDialog(FormOperation.edit, scope.row)">
											编辑
										</el-button>

										<el-button
											text
											:icon="Delete"
											class="red"
											@click="onDelete(scope.$index, scope.row)"
											v-permiss="getPermissKey(ViewFeature.del)">
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
				</ElContainer>
			</ElContainer>
		</ElContainer>
		<modify-diaglog
			:title="table_module.diaglogTitle"
			ref="modifyDiaglogRef"
			@saved="onLoadData"></modify-diaglog>
	</div>
</template>

<script setup lang="ts" name="basetable">
	import { ref, reactive, onMounted } from 'vue';
	import {
		ElMessage,
		ElContainer,
		ElButton,
		ElInput,
		ElMain,
		ElHeader,
		ElTable,
		ElTableColumn,
		ElScrollbar,
		ElPagination,
		ElFooter,
	} from 'element-plus';

	import { Delete, Search, Plus, Setting } from '@element-plus/icons-vue';

	import api from '@/api/sys/menu';
	import modifyDiaglog, {
		type MenuItem as Item,
	} from '@/components/modifyMenu';
	import {
		showLoading,
		closeLoading,
		FormOperation,
		type IPageParam,
		type Table_Module_Base,
		EleConfirm,
		warningArgs,
	} from 'co6co';
	import useSelect from '@/hooks/useMenuSelect';
	import { usePermission, ViewFeature } from '@/hooks/useRoute';
	const { getPermissKey } = usePermission();

	interface IQueryItem extends IPageParam {
		name?: string;
		code?: string;
		parentId?: number;
	}

	interface Table_Module extends Table_Module_Base {
		query: IQueryItem;
		data: Item[];
		currentItem?: Item;
	}
	const table_module = reactive<Table_Module>({
		query: {
			pageIndex: 1,
			pageSize: 10,
			order: 'asc',
			orderBy: '',
		},
		data: [],
		pageTotal: -1,
		diaglogTitle: '',
	});
	const { refresh, getName } = useSelect();

	// 获取表格数据
	const getData = () => {
		showLoading();
		api
			.get_tree_table_svc(table_module.query)
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
	const modifyDiaglogRef = ref<InstanceType<typeof modifyDiaglog>>();
	const onOpenDialog = (operation: FormOperation, row?: Item) => {
		table_module.diaglogTitle =
			operation == FormOperation.add ? '增加新菜单' : '编辑菜单';
		table_module.currentItem = row;
		modifyDiaglogRef.value?.openDialog(operation, row);
	};
	const onLoadData = () => {
		refresh();
		getData();
		modifyDiaglogRef.value?.update();
	};
	// 删除操作
	const onDelete = (index: number, row: Item) => {
		EleConfirm(`确定要删除"${row.name}"吗？`, { ...warningArgs })
			.then(() => {
				showLoading();
				api
					.del_svc(row.id)
					.then((res) => {
						ElMessage.success(res.message || '删除成功'), onLoadData();
					})
					.finally(() => {
						closeLoading();
					});
			})
			.catch(() => {});
	};
	onMounted(() => {
		onLoadData();
	});
</script>
