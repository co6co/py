import { defineComponent, VNodeChild } from 'vue';
import { ref, reactive, onMounted } from 'vue';
import { ElButton, ElInput, ElTableColumn, ElTag } from 'element-plus';
import { Search, Plus, Delete, Edit } from '@element-plus/icons-vue';

import { routeHook } from '@/hooks';
import { tableScope } from '@/constants';

import { TableView, type TableViewInstance } from '@/components/table';
import { useState, useCategory } from '@/hooks/useUserSelect';

import { EnumSelect, FormOperation } from 'co6co';
import useDelete from '@/hooks/useDelete';

import { Associated as AssociatedDiaglog } from 'co6co';
import api, { association_service as ass_api } from '@/api/sys/user';
import ModifyDiaglog, {
	type UserItem as Item,
	ModifyUserInstance,
} from '@/components/modifyUser';
import ResetPwdDiaglog from '@/components/resetPwd';
export default defineComponent({
	name: 'UserTable',
	setup(prop, ctx) {
		//:define
		interface IQueryItem {
			name?: string;
			category?: number;
			state?: number;
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
		const { loadData, selectData, getName, getTagType } = useState();
		const userCategoryHook = useCategory();
		//end use
		//:page
		const viewRef = ref<TableViewInstance>();

		const onSearch = () => {
			viewRef.value?.search();
		};
		const onRefesh = () => {
			viewRef.value?.refesh();
		};

		onMounted(async () => {
			await loadData();
			await userCategoryHook.loadData();
			onSearch();
		});

		//增加/修改

		const diaglogRef = ref<ModifyUserInstance>();
		const onOpenDialog = (operation: FormOperation, row?: Item) => {
			DATA.title =
				operation == FormOperation.add
					? '增加用户'
					: `编辑[${row?.userName}]用户`;
			DATA.currentItem = row;
			diaglogRef.value?.openDialog(operation, row);
		};
		//重置密码
		const resetPwdDiaglogRef = ref<InstanceType<typeof ResetPwdDiaglog>>();
		const onOpenResetDialog = (_: number, row?: Item) => {
			DATA.currentItem = row;
			if (!row) {
				return;
			}
			resetPwdDiaglogRef.value?.openDialog(row?.category!, row);
		};
		//删除
		const { deleteSvc } = useDelete(api.del_svc, onRefesh);
		const onDelete = (_: number, row: Item) => {
			deleteSvc(row.id, row.userName);
		};

		//关联
		const associatedDiaglogRef = ref<InstanceType<typeof AssociatedDiaglog>>();
		const onOpenAssDiaglog = (row?: Item) => {
			DATA.currentItem = row;
			associatedDiaglogRef.value?.openDialog(
				row!.id,
				ass_api.get_association_svc,
				ass_api.save_association_svc
			);
		};
		//:page reader
		const rander = (): VNodeChild => {
			return (
				<TableView dataApi={api.get_table_svc} ref={viewRef} query={DATA.query}>
					{{
						header: () => (
							<>
								<div class="handle-box">
									<ElInput
										clearable
										style={DATA.headItemWidth}
										v-model={DATA.query.name}
										placeholder="名称"
									/>
									<EnumSelect
										clearable
										style={DATA.headItemWidth}
										data={userCategoryHook.selectData.value}
										v-model={DATA.query.category}
										placeholder="用户类型"
									/>
									<EnumSelect
										clearable
										data={selectData.value}
										style={DATA.headItemWidth}
										v-model={DATA.query.state}
										placeholder="状态"
									/>

									<ElButton type="primary" icon={Search} onClick={onSearch}>
										搜索
									</ElButton>
									<ElButton
										type="primary"
										icon={Plus}
										v-permiss={getPermissKey(routeHook.ViewFeature.add)}
										onClick={() => {
											onOpenDialog(FormOperation.add);
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
									prop="userName"
									label="用户名"
									align="center"
									width={180}
									sortable="custom"
									showOverflowTooltip={true}
								/>
								<ElTableColumn
									prop="category"
									label="用户类型"
									align="center"
									width={180}
									sortable="custom"
									showOverflowTooltip={true}>
									{{
										default: (scope: tableScope<Item>) =>
											userCategoryHook.getName(scope.row.category),
									}}
								</ElTableColumn>
								<ElTableColumn
									prop="groupName"
									label="所属用户组"
									align="center"
									sortable="custom"
									showOverflowTooltip={true}
								/>

								<ElTableColumn label="状态" width={80} align="center">
									{{
										default: (scope: tableScope<Item>) => (
											<ElTag type={getTagType(scope.row.state)}>
												{getName(scope.row.state)}
											</ElTag>
										),
									}}
								</ElTableColumn>
								<ElTableColumn
									prop="createTime"
									label="注册时间"
									sortable="custom"
									width={160}
									show-overflow-tooltip={true}
								/>

								<ElTableColumn
									label="操作"
									width={440}
									align="center"
									fixed="right">
									{{
										default: (scope: tableScope<Item>) => (
											<>
												<ElButton
													text={true}
													icon={Edit}
													onClick={() =>
														onOpenDialog(FormOperation.edit, scope.row)
													}
													v-permiss={getPermissKey(routeHook.ViewFeature.edit)}>
													编辑
												</ElButton>
												<ElButton
													text={true}
													icon={Edit}
													onClick={() => onOpenAssDiaglog(scope.row)}
													v-permiss={getPermissKey(
														routeHook.ViewFeature.associated
													)}>
													关联
												</ElButton>
												<ElButton
													text={true}
													icon={Delete}
													onClick={() => onDelete(scope.$index, scope.row)}
													v-permiss={getPermissKey(routeHook.ViewFeature.del)}>
													删除
												</ElButton>
												<ElButton
													text={true}
													icon={Edit}
													onClick={() =>
														onOpenResetDialog(scope.$index, scope.row)
													}
													v-permiss={getPermissKey(
														routeHook.ViewFeature.reset
													)}>
													重置密码
												</ElButton>
											</>
										),
									}}
								</ElTableColumn>
							</>
						),
						footer: () => (
							<>
								<ModifyDiaglog
									ref={diaglogRef}
									title={DATA.title}
									onSaved={onRefesh}
									style="width: 40%"
								/>

								<ResetPwdDiaglog
									ref={resetPwdDiaglogRef}
									pwdShowType={true}
									title="重置密码"
									style="width: 40%"
								/>

								<AssociatedDiaglog
									ref={associatedDiaglogRef}
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
