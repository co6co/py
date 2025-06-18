import { defineComponent, VNodeChild, ref, reactive, onMounted } from 'vue';

import { ElButton, ElInput, ElTableColumn } from 'element-plus';
import { Search, Plus, Delete, Edit, Setting } from '@element-plus/icons-vue';

import {
	FormOperation,
	Associated as AssDiaglog,
	AssociatedInstance as AssDiaglogInstance,
} from 'co6co';
import { routeHook } from '@/hooks';
import { tableScope, ViewFeature } from '@/constants';

import { TableView, type TableViewInstance } from '@/components/table';

import useUserGroupSelect from '@/hooks/useUserGroupSelect';
import Diaglog, {
	type UserGroupItem as Item,
	type ModifyUserGroupInstance as DiaglogInstance,
} from '@/components/modifyUserGroup';
import api, { association_service as ass_api } from '@/api/sys/userGroup';
import useDelete from '@/hooks/useDelete';

export const ViewFeatures = {
	add: ViewFeature.add,
	edit: ViewFeature.edit,
	del: ViewFeature.del,
	associated: ViewFeature.associated,
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
		const { refresh, getName } = useUserGroupSelect();
		//end use
		//:page
		const viewRef = ref<TableViewInstance>();
		const diaglogRef = ref<DiaglogInstance>();
		const onOpenDialog = (row?: Item) => {
			DATA.title = row ? `编辑[${row?.name}]用户组` : '增加用户组';
			DATA.currentItem = row;
			diaglogRef.value?.openDialog(
				row ? FormOperation.edit : FormOperation.add,
				row
			);
		};
		//关联权限
		const assDiaglogRef = ref<AssDiaglogInstance>();
		const onOpenAssDiaglog = (row?: Item) => {
			DATA.currentItem = row;
			assDiaglogRef.value?.openDialog(
				row!.id,
				ass_api.get_association_svc,
				ass_api.save_association_svc
			);
		};
		const onSearch = () => {
			viewRef.value?.search();
		};
		const onRefesh = () => {
			refresh().then(() => {});
			viewRef.value?.refesh();
			diaglogRef.value?.update();
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
					ref={viewRef}
					query={DATA.query}
					row-key="id"
					treeProps={{ children: 'children' }}>
					{{
						header: () => (
							<>
								<div class="handle-box">
									<ElInput
										style={DATA.headItemWidth}
										clearable
										v-model={DATA.query.name}
										placeholder="组名"
									/>
									<ElInput
										style={DATA.headItemWidth}
										clearable
										v-model={DATA.query.code}
										placeholder="编码"
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
									width={315}
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
													icon={Setting}
													onClick={() => onOpenAssDiaglog(scope.row)}
													v-permiss={getPermissKey(ViewFeature.associated)}>
													关联角色
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
								<AssDiaglog
									ref={assDiaglogRef}
									checkStrictly={true}
									style="width: 30%"
									title="关联角色"
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
