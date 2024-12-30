<template>
	<div class="container-layout c-container">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						v-model="table_module.query.name"
						placeholder="名称"
						class="handle-input mr10"></el-input>
					<el-input
						v-model="table_module.query.code"
						placeholder="编码"
						class="handle-input mr10"></el-input>
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
						@sort-change="onColChange2"
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
							:show-overflow-tooltip="true">
							<template #default="scope">
								<router-link
									v-permiss="getPermissKey(ViewFeature.view)"
									:to="{ path: getsubViewPath(scope.row.id) }"
									>{{ scope.row.name }}</router-link
								>
								<span v-non-permiss="getPermissKey(ViewFeature.view)">
									{{ scope.row.name }}</span
								>
							</template>
						</el-table-column>
						<el-table-column
							prop="code"
							label="编码"
							sortable="custom"
							:show-overflow-tooltip="true"></el-table-column>

						<el-table-column
							prop="state"
							label="状态"
							sortable="custom"
							:show-overflow-tooltip="true">
							<template #default="scope">
								<el-tag :type="getTagType(scope.row.state)">
									{{ getName(scope.row.state) }}
								</el-tag>
							</template>
						</el-table-column>

						<el-table-column
							prop="createTime"
							label="创建时间"
							sortable="custom"
							:show-overflow-tooltip="true"></el-table-column>
						<el-table-column
							prop="updateTime"
							label="更新时间"
							sortable="custom"
							:show-overflow-tooltip="true"></el-table-column>
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
					<Pagination
						:option="table_module.query"
						:total="table_module.pageTotal"
						@current-page-change="getData"
						@size-chage="getData" />
				</div>
			</el-footer>
		</el-container>
		<!-- 弹出框 -->
		<modifyDiaglog
			@saved="onSearch"
			ref="modifyDiaglogRef"
			:title="table_module.diaglogTitle" />
	</div>
</template>

<script setup lang="ts" name="basetable">
	import { ref, reactive, onMounted } from 'vue';
	import {
		ElInput,
		ElButton,
		ElHeader,
		ElTableColumn,
		ElTag,
		ElTable,
		ElScrollbar,
		ElMain,
		ElFooter,
		ElContainer,
	} from 'element-plus';
	import { Delete, Edit, Search, Plus } from '@element-plus/icons-vue';
	import { dictTypeSvc as svc } from '@/api/dict';
	import { usePermission, ViewFeature, useRouteData } from '@/hooks/useRoute';
	import { useState } from '@/hooks/useDictState';

	import { replaceRouteParams } from '@/utils';

	const { getPermissKey } = usePermission();
	const { loadData, getName, getTagType } = useState();
	import modifyDiaglog, {
		type DictTypeItem as Item,
		type ModifyDictTypeInstance,
	} from '@/components/modifyDictType';

	import {
		showLoading,
		closeLoading,
		FormOperation,
		onColChange,
		Pagination,
		type IPageParam,
		type Table_Module_Base,
	} from 'co6co';
	import { getViewPath } from '@/views';
	import useDelete from '@/hooks/useDelete';

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

	const onColChange2 = (column: any) => {
		onColChange(column, table_module.query, getData);
	};

	const modifyDiaglogRef = ref<ModifyDictTypeInstance>();

	//编辑角色
	const onOpenDialog = (operation: FormOperation, row?: Item) => {
		table_module.diaglogTitle =
			operation == FormOperation.add
				? '增加字典类型'
				: `编辑[${row?.name}]菜单`;
		table_module.currentItem = row;
		modifyDiaglogRef.value?.openDialog(operation, row);
	};

	const { deleteSvc } = useDelete(svc.del_svc, getData);
	const onDelete = (_: number, row: Item) => {
		deleteSvc(row.id, row.name);
	};
	const subViewPath = ref('');
	const getsubViewPath = (dictTypeId: number) => {
		return replaceRouteParams(subViewPath.value, { id: dictTypeId.toString() });
	};

	onMounted(async () => {
		await loadData();
		const { queryRouteItem } = useRouteData();
		const componentName = getViewPath('DictView');
		const routeItem = queryRouteItem((d) => {
			return d.component == componentName;
		});
		if (routeItem) {
			subViewPath.value = routeItem.url;
		}
		getData();

		console.warn('子视图KEY', getPermissKey(ViewFeature.view));
	});
</script>
<style scoped>
	.cell .el-button {
		padding: 8px 7px;
	}
</style>
