import { defineComponent, VNodeChild, ref, reactive } from 'vue';

import { ElButton, ElInput, ElTableColumn } from 'element-plus';
import { Search, Plus, Delete, Edit, Setting } from '@element-plus/icons-vue';

import {
	FormOperation,
	Associated as AssDiaglog,
	AssociatedInstance as AssDiaglogInstance,
} from 'co6co';
import { routeHook } from '@/hooks';
import { tableScope } from '@/constants';

import { TableView, type TableViewInstance } from '@/components/table';

import Diaglog, {
	type RoleItem as Item,
	type ModifyRoleInstance as DiaglogInstance,
} from '@/components/modifyRole';
import api, { association_service as ass_api } from '@/api/sys/role';
import useDelete from '@/hooks/useDelete';
export default defineComponent({
	setup(prop, ctx) {
		//:define
		interface IQueryItem {
			name?: string;
			code?: string;
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
		//end use
		//:page
		const viewRef = ref<TableViewInstance>();
		const diaglogRef = ref<DiaglogInstance>();
		const onOpenDialog = (row?: Item) => {
			DATA.title = row ? `编辑[${row?.name}]角色` : '增加角色';
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
			viewRef.value?.refesh();
		};

		const { deleteSvc } = useDelete(api.del_svc, onRefesh);
		const onDelete = (_: number, row: Item) => {
			deleteSvc(row.id, row.name);
		};

		//:page reader
		const rander = (): VNodeChild => {
			return (
				<TableView
					dataApi={api.get_table_svc}
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
										placeholder="角色名"
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
										v-permiss={getPermissKey(routeHook.ViewFeature.add)}
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
								<ElTableColumn label="序号" width={55} align="center">
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
									width={340}
									align="center"
									fixed="right">
									{{
										default: (scope: tableScope<Item>) => (
											<>
												<ElButton
													text={true}
													icon={Edit}
													onClick={() => onOpenDialog(scope.row)}
													v-permiss={getPermissKey(routeHook.ViewFeature.edit)}>
													编辑
												</ElButton>
												<ElButton
													text={true}
													icon={Setting}
													onClick={() => onOpenAssDiaglog(scope.row)}
													v-permiss={getPermissKey(
														routeHook.ViewFeature.associated
													)}>
													权限设置
												</ElButton>
												<ElButton
													text={true}
													icon={Delete}
													onClick={() => onDelete(scope.$index, scope.row)}
													v-permiss={getPermissKey(routeHook.ViewFeature.del)}>
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
									title="权限设置"
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
