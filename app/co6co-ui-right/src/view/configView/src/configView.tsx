import { defineComponent, VNodeChild } from 'vue';
import { ref, reactive, onMounted } from 'vue';
import { ElButton, ElInput, ElTableColumn } from 'element-plus';
import { Search, Plus, Delete, Edit } from '@element-plus/icons-vue';

import { FormOperation } from 'co6co';
import { routeHook, tableScope } from 'co6co-right';
import { TableView, type TableViewInstance } from '@/components/table';
import { configSvc as svc } from '@/api/config';
import useDelete from '@/hooks/useDelete';

import Diaglog, {
	type ConfigItem as Item,
	type MdifyConfigInstance,
} from '@/components/modifyConfig';
import { configSvc as api } from '@/api/config';
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
		}>({
			query: {},
		});

		//:use
		const { getPermissKey } = routeHook.usePermission();

		//end use
		//:page
		const viewRef = ref<TableViewInstance>();
		const diaglogRef = ref<MdifyConfigInstance>();

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

		const { deleteSvc } = useDelete(svc.del_svc, onRefesh);
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
										clearable
										v-model={DATA.query.name}
										placeholder="名称"
										class="handle-input"
									/>
									<ElInput
										clearable
										v-model={DATA.query.code}
										placeholder="编码"
										class="handle-input"
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
									show-overflow-tooltip={true}></ElTableColumn>
								<ElTableColumn
									prop="updateTime"
									label="更新时间"
									sortable="custom"
									width={160}
									show-overflow-tooltip={true}></ElTableColumn>
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
									onSaved={onRefesh}></Diaglog>
							</>
						),
					}}
				</TableView>
			);
		};
		return rander;
	}, //end setup
});
