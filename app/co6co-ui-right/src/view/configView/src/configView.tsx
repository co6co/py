import { defineComponent, VNodeChild } from 'vue';
import { ref, reactive, onMounted } from 'vue';
import { ElButton, ElInput, ElTableColumn } from 'element-plus';
import { Search, Plus, Delete, Edit } from '@element-plus/icons-vue';

import { FormOperation } from 'co6co';
import { routeHook } from '@/hooks';
import { tableScope, ViewFeature } from '@/constants';

import { TableView, type TableViewInstance } from '@/components/table';
import useDelete from '@/hooks/useDelete';

import Diaglog, {
	type ConfigItem as Item,
	type MdifyConfigInstance as DiaglogInstance,
} from '@/components/modifyConfig';
import { configSvc as api } from '@/api/config';

export const ViewFeatures = {
	add: ViewFeature.add,
	edit: ViewFeature.edit,
	del: ViewFeature.del,
	sched: ViewFeature.sched,
	stop: ViewFeature.stop,
	execute: ViewFeature.execute,
};
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
			DATA.title = row ? `编辑[${row?.name}]` : '增加';
			DATA.currentItem = row;
			diaglogRef.value?.openDialog(
				row ? FormOperation.edit : FormOperation.add,
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
			onSearch();
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
										placeholder="名称"
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
									label="编码"
									align="center"
									sortable="custom"
									showOverflowTooltip={true}
								/>
								<ElTableColumn
									prop="value"
									label="配置"
									sortable="custom"
									showOverflowTooltip={true}
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
							<Diaglog ref={diaglogRef} title={DATA.title} onSaved={onRefesh} />
						),
					}}
				</TableView>
			);
		};
		return rander;
	}, //end setup
});
