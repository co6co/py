<template>
	<div class="container-layout c-container">
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
					<el-button
						v-if="allowBatch"
						v-permiss="getPermissKey(ViewFeature.add)"
						type="primary"
						:icon="Plus"
						@click="onOpenBatchDialog(FormOperation.add)"
						>批量新增</el-button
					>
				</div>
			</el-header>
			<ElContainer>
				<ElContainer>
					<el-main>
						<el-scrollbar>
							<el-table
								:data="table_module.data"
								@sort-change="onColChange2"
								border
								class="table"
								highlight-current-row
								header-cell-class-name="table-header"
								@current-change="onCurrentChange"
								row-key="id"
								:tree-props="{ children: 'children' }">
								<el-table-column fixed label="序号" width="100" align="center">
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
							<Pagination
								:option="table_module.query"
								:total="table_module.pageTotal"
								@current-page-change="getData"
								@size-chage="getData" />
						</div>
					</el-footer>
				</ElContainer>
			</ElContainer>
		</ElContainer>
		<modify-diaglog
			:title="table_module.diaglogTitle"
			ref="modifyDiaglogRef"
			@saved="onLoadData" />
		<BatchAddMenu
			:title="table_module.diaglogTitle"
			ref="batchAddMenuRef"
			@saved="onLoadData" />
	</div>
</template>

<script setup lang="ts" name="basetable">
	import { ref, reactive, onMounted, computed } from 'vue';
	import {
		ElContainer,
		ElButton,
		ElInput,
		ElMain,
		ElHeader,
		ElTable,
		ElTableColumn,
		ElScrollbar,
		ElFooter,
	} from 'element-plus';

	import { Delete, Search, Plus, Setting } from '@element-plus/icons-vue';

	import svc from '@/api/sys/menu';
	import modifyDiaglog, {
		BatchAddMenu,
		type MenuItem as Item,
	} from '@/components/modifyMenu';

	import {
		showLoading,
		closeLoading,
		FormOperation,
		onColChange,
		Pagination,
		type IPageParam,
		type Table_Module_Base,
	} from 'co6co';
	import useSelect, { MenuCateCategory } from '@/hooks/useMenuSelect';
	import { usePermission, ViewFeature } from '@/hooks/useRoute';
	import useDelete from '@/hooks/useDelete';

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
		svc
			.get_tree_table_svc(table_module.query)
			.then((res) => {
				table_module.data = res.data;
				table_module.pageTotal = res.total || (res.data ? res.data.length : 0);
			})
			.finally(() => {
				closeLoading();
			});
	};

	// 查询操作
	const onSearch = () => {
		getData();
	};

	const onColChange2 = (column: any) => {
		onColChange(column, table_module.query, getData);
	};
	const modifyDiaglogRef = ref<InstanceType<typeof modifyDiaglog>>();

	const onOpenDialog = (operation: FormOperation, row?: Item) => {
		table_module.diaglogTitle =
			operation == FormOperation.add ? '增加新菜单' : '编辑菜单';
		if (operation == FormOperation.edit) table_module.currentItem = row;
		modifyDiaglogRef.value?.openDialog(operation, table_module.currentItem);
	};
	const batchAddMenuRef = ref<InstanceType<typeof BatchAddMenu>>();
	const onOpenBatchDialog = (operation: FormOperation) => {
		table_module.diaglogTitle = '批量增加';

		batchAddMenuRef.value?.openDialog(table_module.currentItem!);
	};
	const onLoadData = async () => {
		await refresh();
		getData();
		modifyDiaglogRef.value?.update();
	};
	// 删除操作
	const { deleteSvc } = useDelete(svc.del_svc, getData);
	const onDelete = (_: number, row: Item) => {
		deleteSvc(row.id, row.name);
	};
	onMounted(() => {
		onLoadData();
	});

	const onCurrentChange = (currentRow: Item, oldCurrentRow: Item) => {
		table_module.currentItem = currentRow;
	};
	const allowBatch = computed(() => {
		return (
			table_module.currentItem &&
			(table_module.currentItem.category == MenuCateCategory.SubVIEW ||
				table_module.currentItem.category == MenuCateCategory.VIEW)
		);
	});
</script>
