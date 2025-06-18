import {
	defineComponent,
	VNodeChild,
	ref,
	reactive,
	computed,
	onMounted,
} from 'vue';

import { ElButton, ElInput, ElTableColumn } from 'element-plus';
import { Search, Plus, Delete, Edit } from '@element-plus/icons-vue';

import { FormOperation } from 'co6co';
import { routeHook } from '@/hooks';
import { tableScope, ViewFeature } from '@/constants';
import useSelect, { MenuCateCategory } from '@/hooks/useMenuSelect';
import { TableView, type TableViewInstance } from '@/components/table';

import Diaglog, {
	type MenuItem as Item,
	BatchAddMenu as BatchDiaglog,
	type ModifyMenuInstance as DiaglogInstance,
	type BatchAddMenuInstance as BatchInstance,
} from '@/components/modifyMenu';
import api from '@/api/sys/menu';
import useDelete from '@/hooks/useDelete';

export const ViewFeatures = {
	add: ViewFeature.add,
	edit: ViewFeature.edit,
	del: ViewFeature.del,
};
export default defineComponent({
	setup(prop, ctx) {
		//:define
		interface IQueryItem {
			name?: string;
			code?: string;
			parentId?: number;
		}
		const DATA = reactive<{
			title?: string;
			query: IQueryItem;
			currentItem?: Item;
			headItemWidth: { width: string };
		}>({
			query: {},
			headItemWidth: { width: '180px' },
		});

		//:use
		const { getPermissKey } = routeHook.usePermission();
		const { refresh, getName } = useSelect();
		//end use
		//:page
		const viewRef = ref<TableViewInstance>();
		const diaglogRef = ref<DiaglogInstance>();
		const onOpenDialog = (row?: Item) => {
			DATA.title = row ? `编辑[${row?.name}]菜单` : '增加新菜单';
			DATA.currentItem = row;
			diaglogRef.value?.openDialog(
				row ? FormOperation.edit : FormOperation.add,
				row
			);
		};

		//批量添加
		const batchDiaglogRef = ref<BatchInstance>();
		const onOpenBatchDialog = () => {
			DATA.title = '批量增加';
			batchDiaglogRef.value?.openDialog(DATA.currentItem!);
		};
		const onCurrentChange = (currentRow: Item, oldCurrentRow: Item) => {
			DATA.currentItem = currentRow;
		};
		const allowBatch = computed(() => {
			return (
				DATA.currentItem &&
				(DATA.currentItem.category == MenuCateCategory.SubVIEW ||
					DATA.currentItem.category == MenuCateCategory.VIEW)
			);
		});
		const onSearch = () => {
			viewRef.value?.search();
		};
		const onRefesh = async () => {
			viewRef.value?.refesh();
			await refresh();
		};

		const { deleteSvc } = useDelete(api.del_svc, onRefesh);
		const onDelete = (_: number, row: Item) => {
			deleteSvc(row.id, row.name);
		};
		onMounted(async () => {
			await refresh();
		});
		//:page reader
		const rander = (): VNodeChild => {
			return (
				<TableView
					dataApi={api.get_tree_table_svc}
					highlightCurrentRow={true}
					ref={viewRef}
					query={DATA.query}
					row-key="id"
					onCurrentChange={onCurrentChange}
					treeProps={{ children: 'children' }}>
					{{
						header: () => (
							<>
								<div class="handle-box">
									<ElInput
										style={DATA.headItemWidth}
										clearable
										v-model={DATA.query.name}
										placeholder="菜单名称"
									/>
									<ElInput
										style={DATA.headItemWidth}
										clearable
										v-model={DATA.query.code}
										placeholder="菜单编码"
									/>

									<ElButton type="primary" icon={Search} onClick={onSearch}>
										搜索
									</ElButton>
									<ElButton
										type="primary"
										icon={Plus}
										v-permiss={getPermissKey(ViewFeature.add)}
										onClick={() => {
											onOpenDialog();
										}}>
										新增
									</ElButton>
									{allowBatch.value ? (
										<ElButton
											type="primary"
											icon={Plus}
											v-permiss={getPermissKey(ViewFeature.add)}
											onClick={() => {
												onOpenBatchDialog();
											}}>
											批量新增
										</ElButton>
									) : (
										<></>
									)}
								</div>
							</>
						),
						default: () => (
							<>
								<ElTableColumn label="序号" width={110} align="center">
									{{
										default: (scope: tableScope) =>
											viewRef.value?.rowIndex(scope.$index),
									}}
								</ElTableColumn>
								<ElTableColumn
									prop="name"
									label="名称"
									align="center"
									width={180}
									sortable="custom"
									showOverflowTooltip={true}
								/>
								<ElTableColumn
									prop="code"
									width={120}
									label="父节点"
									align="center"
									sortable="custom"
									showOverflowTooltip={true}>
									{{
										default: (scope: tableScope<Item>) => {
											return getName(scope.row.parentId) || '-';
										},
									}}
								</ElTableColumn>
								<ElTableColumn
									prop="code"
									width={120}
									label="代码"
									align="center"
									sortable="custom"
									showOverflowTooltip={true}
								/>
								<ElTableColumn
									prop="order"
									width={80}
									label="排序"
									align="center"
									sortable="custom"
								/>

								<ElTableColumn
									prop="createTime"
									label="创建时间"
									sortable="custom"
									width={160}
									show-overflow-tooltip={true}
								/>
								<ElTableColumn
									prop="updateTime"
									label="更新时间"
									sortable="custom"
									width={160}
									show-overflow-tooltip={true}
								/>
								<ElTableColumn
									label="操作"
									width={222}
									align="center"
									fixed="right">
									{{
										default: (scope: tableScope<Item>) => (
											<>
												<ElButton
													text={true}
													icon={Edit}
													onClick={() => onOpenDialog(scope.row)}
													v-permiss={getPermissKey(ViewFeature.edit)}>
													编辑
												</ElButton>

												<ElButton
													text={true}
													icon={Delete}
													onClick={() => onDelete(scope.$index, scope.row)}
													v-permiss={getPermissKey(ViewFeature.del)}>
													删除
												</ElButton>
											</>
										),
									}}
								</ElTableColumn>
							</>
						),
						footer: () => (
							<>
								<Diaglog
									ref={diaglogRef}
									title={DATA.title}
									onSaved={onRefesh}
								/>
								<BatchDiaglog
									ref={batchDiaglogRef}
									checkStrictly={true}
									onSaved={onRefesh}
									title={DATA.title}
								/>
							</>
						),
					}}
				</TableView>
			);
		};
		return rander;
	}, //end setup
});
