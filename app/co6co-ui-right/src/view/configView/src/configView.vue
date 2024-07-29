<template>
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						v-model="table_module.query.name"
						placeholder="名称"
						class="handle-input"></el-input>
					<el-input
						v-model="table_module.query.code"
						placeholder="编码"
						class="handle-input"></el-input>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>搜索</el-button
					>
					<el-button
						type="primary"
						v-permiss="getPermissKey(ViewFeature.add)"
						:icon="Plus"
						@click="onOpenDialog(0)"
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
						<el-table-column
							label="序号"
							width="55"
							align="center"
							:show-overflow-tooltip="true">
							<template #default="scope"> {{ scope.$index + 1 }} </template>
						</el-table-column>

						<el-table-column
							prop="name"
							label="名称"
							sortable="custom"
							:show-overflow-tooltip="true" />
						<el-table-column
							prop="code"
							label="编码"
							sortable="custom"
							:show-overflow-tooltip="true" />

						<el-table-column
							prop="value"
							label="配置"
							sortable
							:show-overflow-tooltip="true" />

						<el-table-column
							prop="createTime"
							label="创建时间"
							sortable="custom"
							:show-overflow-tooltip="true" />
						<el-table-column
							prop="updateTime"
							label="更新时间"
							sortable="custom"
							:show-overflow-tooltip="true" />
						<el-table-column
							label="操作"
							width="180"
							fixed="right"
							align="center">
							<template #default="scope">
								<el-button
									text
									:icon="Edit"
									v-permiss="getPermissKey(ViewFeature.edit)"
									@click="onOpenDialog(1, scope.row)">
									编辑
								</el-button>
								<el-button
									text
									:icon="Delete"
									class="red"
									v-permiss="getPermissKey(ViewFeature.del)"
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
		<modifyDiaglog @saved="onSearch" ref="modifyDiaglogRef" title="编辑" />
	</div>
</template>

<script setup lang="ts" name="basetable">
	import { ref, reactive, onMounted } from 'vue';
	import {
		ElInput,
		ElButton,
		ElHeader,
		ElTableColumn,
		ElTable,
		ElScrollbar,
		ElMain,
		ElPagination,
		ElFooter,
		ElContainer,
	} from 'element-plus';
	import { Delete, Edit, Search, Plus } from '@element-plus/icons-vue';
	import { configSvc as svc } from '@/api/config';
	import { usePermission, ViewFeature } from '@/hooks/useRoute';
	import useDelete from '@/hooks/useDelete';

	const { getPermissKey } = usePermission();
	import modifyDiaglog, {
		type ConfigItem as Item,
		type MdifyConfigInstance,
	} from '@/components/modifyConfig';

	import {
		showLoading,
		closeLoading,
		FormOperation,
		type IPageParam,
		type Table_Module_Base,
	} from 'co6co';
	interface IQueryItem extends IPageParam {
		name?: string;
		code?: string;
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
		svc
			.get_table_svc(table_module.query)
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

	const modifyDiaglogRef = ref<MdifyConfigInstance>();

	//编辑角色
	const onOpenDialog = (operation: FormOperation, row?: Item) => {
		table_module.diaglogTitle =
			operation == FormOperation.add ? '增加配置' : `编辑[${row?.name}]配置`;
		table_module.currentItem = row;
		modifyDiaglogRef.value?.openDialog(operation, row);
	};

	const { deleteSvc } = useDelete(svc.del_svc, getData);
	const onDelete = (_: number, row: Item) => {
		deleteSvc(row.id, row.name);
	};
	onMounted(() => {
		getData();
	});
</script>
<style scoped>
	.cell .el-button {
		padding: 8px 7px;
	}
</style>
