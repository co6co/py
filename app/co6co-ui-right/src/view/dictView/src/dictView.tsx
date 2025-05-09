import { defineComponent, VNodeChild, ref, reactive, onMounted } from 'vue';

import { useRoute, useRouter } from 'vue-router';
import {
	ElButton,
	ElInput,
	ElTableColumn,
	ElMessage,
	ElTag,
} from 'element-plus';
import { Search, Plus, Delete, Edit } from '@element-plus/icons-vue';

import { FormOperation } from 'co6co';
import { routeHook } from '@/hooks';
import { useState } from '@/hooks/useDictState';
import { tableScope } from '@/constants';

import { TableView, type TableViewInstance } from '@/components/table';

import Diaglog, {
	type DictItem as Item,
	type ModifyDictInstance as DiaglogInstance,
} from '@/components/modifyDict';
import { dictSvc as api } from '@/api/dict';
import useDelete from '@/hooks/useDelete';
export default defineComponent({
	setup(prop, ctx) {
		//:define
		interface IQueryItem {
			dictTypeId: number;
			name?: string;
			code?: string;
		}
		const DATA = reactive<{
			title?: string;
			query: IQueryItem;
			currentItem?: Item;
			headItemWidth: { width: string };
		}>({
			query: { dictTypeId: 0 },
			headItemWidth: { width: '180px' },
		});

		//:use
		const { getPermissKey } = routeHook.usePermission();
		const { loadData, getName, getTagType } = useState();
		const route = useRoute();
		const router = useRouter();
		//end use
		//:page
		const viewRef = ref<TableViewInstance>();
		const diaglogRef = ref<DiaglogInstance>();

		const onOpenDialog = (row?: Item) => {
			DATA.title = row ? `编辑[${row?.name}]字典` : '增加字典';
			DATA.currentItem = row;
			diaglogRef.value?.openDialog(
				row ? FormOperation.edit : FormOperation.add,
				DATA.query.dictTypeId,
				row
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

		onMounted(async () => {
			await loadData();
			let id = router.currentRoute.value.params.id || route.query.id;
			if (id) {
				DATA.query.dictTypeId = Number(id);
				onSearch();
			} else {
				ElMessage.error('参数不正确！');
			}
		});

		//:page reader
		const rander = (): VNodeChild => {
			return (
				<TableView dataApi={api.get_table_svc} ref={viewRef} query={DATA.query}>
					{{
						header: () => (
							<>
								<div class="handle-box">
									<ElInput
										style={DATA.headItemWidth}
										clearable
										v-model={DATA.query.name}
										placeholder="字典名称"
									/>
									<ElInput
										style={DATA.headItemWidth}
										clearable
										v-model={DATA.query.code}
										placeholder="字典编码"
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
									prop="value"
									label="值"
									align="center"
									sortable="custom"
									showOverflowTooltip={true}
								/>
								<ElTableColumn
									prop="value"
									label="配置"
									sortable="custom"
									showOverflowTooltip={true}>
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
													v-permiss={getPermissKey(routeHook.ViewFeature.edit)}>
													编辑
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
							</>
						),
					}}
				</TableView>
			);
		};
		return rander;
	}, //end setup
});
