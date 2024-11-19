<template>
	<div class="container-layout c-container">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						style="width: 160px"
						clearable
						v-model="table_module.query.name"
						placeholder="角色名"
						class="handle-input mr10"></el-input>
					<el-input
						style="width: 160px"
						clearable
						v-model="table_module.query.code"
						placeholder="编码"
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
			<el-container>
				<el-container>
					<el-main>
						<el-scrollbar>
							<el-table
								:data="table_module.data"
								@sort-change="onColChange2"
								border
								class="table"
								header-cell-class-name="table-header"
								row-key="id"
								:tree-props="{ children: 'children' }">
								<el-table-column label="序号" width="100" align="center">
									<template #default="scope"> {{ scope.$index + 1 }} </template>
								</el-table-column>
								<el-table-column
									width="120"
									prop="name"
									label="名称"
									sortable="custom"
									:show-overflow-tooltip="true"></el-table-column>

								<el-table-column
									width="120"
									prop="code"
									label="代码"
									sortable="custom"
									:show-overflow-tooltip="true"></el-table-column>

								<el-table-column
									width="80"
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
								<el-table-column label="操作" align="center" fixed="right">
									<template #default="scope">
										<el-button
											v-permiss="getPermissKey(ViewFeature.edit)"
											text
											:icon="Setting"
											@click="onOpenDialog(FormOperation.edit, scope.row)">
											编辑
										</el-button>
										<el-button
											v-permiss="getPermissKey(ViewFeature.associated)"
											text
											:icon="Setting"
											@click="onOpenAssMenuDiaglog(scope.row)">
											权限设置
										</el-button>

										<el-button
											v-permiss="getPermissKey(ViewFeature.del)"
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
							<Pagination
								:option="table_module.query"
								:total="table_module.pageTotal"
								@current-page-change="getData"
								@size-chage="getData" />
						</div>
					</el-footer>
				</el-container>
			</el-container>
		</el-container>

		<modify-diaglog
			:title="table_module.diaglogTitle"
			ref="modifyDiaglogRef"
			@saved="onLoadData"></modify-diaglog>

		<role-ass-menu-diaglog
			:check-strictly="true"
			style="width: 30%"
			title="权限设置"
			ref="roleAssMenuDiaglogRef"></role-ass-menu-diaglog>
	</div>
</template>

<script setup lang="ts" name="basetable">
	import { ref, reactive, onMounted } from 'vue';
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

	import modifyDiaglog, {
		type RoleItem as Item,
	} from '@/components/modifyRole';
	import {
		showLoading,
		closeLoading,
		FormOperation,
		onColChange,
		Pagination,
		Associated as roleAssMenuDiaglog,
		type IPageParam,
		type Table_Module_Base,
	} from 'co6co';
	import svc, { association_service as ass_api } from '@/api/sys/role';
	import { usePermission, ViewFeature } from '@/hooks/useRoute';
	import useDelete from '@/hooks/useDelete';
	const { getPermissKey } = usePermission();

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
			pageIndex: 1,
			pageSize: 10,
			order: 'asc',
			orderBy: '',
		},
		data: [],
		pageTotal: -1,
		diaglogTitle: '',
	});
	// 获取表格数据
	const getData = () => {
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

	//编辑角色
	const modifyDiaglogRef = ref<InstanceType<typeof modifyDiaglog>>();
	const onOpenDialog = (operation: FormOperation, row?: Item) => {
		table_module.diaglogTitle =
			operation == FormOperation.add ? '增加角色' : `编辑[${row?.name}]角色`;
		table_module.currentItem = row;
		modifyDiaglogRef.value?.openDialog(operation, row);
	};
	//关联权限
	const roleAssMenuDiaglogRef = ref<InstanceType<typeof roleAssMenuDiaglog>>();
	const onOpenAssMenuDiaglog = (row?: Item) => {
		table_module.currentItem = row;
		roleAssMenuDiaglogRef.value?.openDialog(
			row!.id,
			ass_api.get_association_svc,
			ass_api.save_association_svc
		);
	};

	const onLoadData = () => {
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
</script>
