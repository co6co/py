import {
	ref,
	reactive,
	onMounted,
	computed,
	defineComponent,
	VNodeChild,
} from 'vue';
import {
	ElButton,
	ElInput,
	ElTableColumn,
	ElRow,
	ElCol,
	ElLink,
	ElTooltip,
} from 'element-plus';
import {
	Search,
	ArrowLeftBold,
	Refresh,
	Delete,
	UploadFilled,
} from '@element-plus/icons-vue';
import style from '@/assets/css/file.module.less';
import { byte2Unit } from 'co6co';
import { routeHook, deleteHook } from '@/hooks';
import { tableScope } from '@/constants';
import { download_header_svc } from '@/api';

import { TableView, TableViewInstance } from '@/components/table';
import { Download } from '@/components/download';
import Diaglog from '@/components/dragUploader';
import {
	list_svc,
	getResourceUrl,
	del_svc,
	list_param,
	batch_del_svc,
	list_res as Item,
} from '@/api/file';

import { ConfigCodes } from '@/constants/config';
import { useConfig } from '@/hooks/useConfig';
import { useKeyUp } from '@/hooks/useKey';

export default defineComponent({
	setup(prop, ctx) {
		const DATA = reactive<{
			query: list_param;
			currentItem?: Item;
			split: RegExp;
			isMask: boolean;
		}>({
			query: { root: 'I:' },
			split: /[/\\]/,
			isMask: false,
		});
		//:use
		const { getPermissKey } = routeHook.usePermission();

		//end use
		//:page
		const viewRef = ref<TableViewInstance>();
		const diaglogRef = ref<InstanceType<typeof Diaglog>>();

		const onOpenDialog = (clearFile: boolean = true) => {
			if (clearFile) diaglogRef.value?.clearFile();
			diaglogRef.value?.openDialog(DATA.query.root);
		};
		const onSearch = () => {
			viewRef.value?.search();
		};
		const onRefesh = () => {
			viewRef.value?.refesh();
		};
		const configHooks = useConfig(ConfigCodes.FILE_MGR_CODE);
		onMounted(async () => {
			await configHooks.loadData();
			DATA.query.root = configHooks.getValue(true);
			onSearch();
			/** 不支持$on .$off 
			viewRef.value?.tableRef!.$on('selection-change', onSelectionChange);

			// 清理事件监听器
			onBeforeUnmount(() => {
				viewRef.value?.tableRef!.$off('selection-change', onSelectionChange);
			});
			 */
		});
		const isEditing = ref(false);
		const handleFocus = () => {
			isEditing.value = true;
		};
		const handleBlur = () => {
			isEditing.value = false;
		};
		const previewRoot = computed(() => {
			if (DATA.query.root) {
				const arr = DATA.query.root.split(DATA.split);
				return '根目录' + arr.join(' > ').substring(1);
			}
			return '';
		});
		const onRootUp = () => {
			if (DATA.query.root) {
				const arr = DATA.query.root.split(DATA.split);
				const result = arr.slice(0, arr.length - 1);
				//console.info(arr, result);
				if (result.length == 1 && result[0] == '') DATA.query.root = '/';
				else DATA.query.root = result.join('/');
				onSearch();
			}
		};
		const ontresultFileter = (data: { res: any[]; root: string }) => {
			DATA.query.root = data.root;
			return data.res;
		};
		const onClickSubFolder = (path: string) => {
			DATA.query.root = path;
			onSearch();
		};
		const onClickClcFolder = (row: Item & { loading?: boolean }) => {
			row.loading = true;
			download_header_svc(getResourceUrl(row.path, true), true)
				.then((res) => {
					row.size = Number(res.headers['content-length']);
				})
				.finally(() => {
					row.loading = false;
				});
		};

		const onDrop = (event) => {
			if (getPermissKey(routeHook.ViewFeature.upload)) {
				diaglogRef.value?.clearFile();
				diaglogRef.value?.onDrop(event);
				DATA.isMask = false;
				if (diaglogRef.value?.hasFile()) {
					onOpenDialog(false);
				}
			}
		};

		const onDragOver = (event) => {
			if (getPermissKey(routeHook.ViewFeature.upload)) {
				DATA.isMask = true;
				diaglogRef.value?.onDragOver(event);
			}
		};
		useKeyUp((event) => {
			if (event.key === 'Escape') DATA.isMask = false;
		});
		const multipleSelection = ref<Item[]>([]);
		const onSelectionChange = (val: Item[]) => {
			multipleSelection.value = val;
		};
		const tableSelected = computed(() => {
			return multipleSelection.value.length > 0;
		});
		//删除
		const { deleteSvc } = deleteHook.default(del_svc, onRefesh);
		const onDelete = (_: number, row: Item) => {
			deleteSvc(row.path, row.name);
		};
		const onPreview = (_: number, row: Item) => {
			console.warn('为实现！');
		};
		const batchDelTip = deleteHook.default(batch_del_svc, onRefesh);
		const onBatchDel = () => {
			const selectPath = multipleSelection.value.map((m) => m.path);
			batchDelTip.deleteSvc2(
				selectPath,
				batchDelTip.createConfirmBox(
					`确定要删除[${selectPath.length}]条数据`,
					'删除文件或文件夹警告',
					'warning'
				)
			);
		};

		//:page reader
		const rander = (): VNodeChild => {
			return (
				<div onDrop={onDrop} onDragover={onDragOver}>
					{DATA.isMask ? (
						<div class={[style['upload-box'], 'el-overlay']}>
							<span>上传文件或文件夹到当前目录</span>
						</div>
					) : (
						<></>
					)}
					<TableView
						dataApi={list_svc}
						ref={viewRef}
						autoLoadData={false}
						query={DATA.query}
						showPaged={false}
						onSelection-change={onSelectionChange}
						resultFilter={ontresultFileter}>
						{{
							header: () => (
								<ElRow>
									<ElCol span={12}>
										<div class="handle-box">
											<ElInput
												style="flex:0 0 70%"
												v-model={DATA.query.root}
												class={isEditing.value ? style.editor : style.show}
												value={
													isEditing.value ? DATA.query.root : previewRoot.value
												}
												onFocus={handleFocus}
												onBlur={handleBlur}
												onChange={onSearch}>
												{{
													prepend: () => (
														<ElButton
															title="上级目录"
															icon={ArrowLeftBold}
															onClick={onRootUp}
														/>
													),
													append: () => (
														<>
															<ElButton
																title="刷新"
																type="primary"
																icon={Refresh}
																onClick={onSearch}
															/>
														</>
													),
												}}
											</ElInput>
											<ElButton
												style="flex:0 0"
												icon={UploadFilled}
												v-permiss={getPermissKey(routeHook.ViewFeature.upload)}
												onClick={() => onOpenDialog()}
												v-slots={{ default: () => '上传' }}
											/>
											{tableSelected.value ? (
												<ElButton
													type="danger"
													icon={Delete}
													v-permiss={getPermissKey(routeHook.ViewFeature.del)}
													onClick={onBatchDel}>
													删除选中
												</ElButton>
											) : (
												<></>
											)}
										</div>
									</ElCol>
									<ElCol span={6} offset={6}>
										<ElInput
											style="width: 160px"
											clearable
											v-model={DATA.query.name}
											placeholder="搜索文件/目录"
											class="handle-input"
										/>
										<ElButton type="primary" icon={Search} onClick={onSearch}>
											搜索
										</ElButton>
									</ElCol>
								</ElRow>
							),
							default: () => (
								<>
									<ElTableColumn type="selection" width={55} />
									<ElTableColumn label="序号" width={112} align="center">
										{{
											default: (scope: tableScope) =>
												viewRef.value?.rowIndex(scope.$index),
										}}
									</ElTableColumn>
									<ElTableColumn
										label="名称"
										prop="name"
										align="center"
										showOverflowTooltip={true}>
										{{
											default: (scope: { row: Item }) => (
												<ElTooltip
													effect="dark"
													content={scope.row.path}
													showAfter={1500}>
													{scope.row.isFile ? (
														scope.row.name
													) : (
														<ElLink
															onClick={() => onClickSubFolder(scope.row.path)}>
															{scope.row.name}
														</ElLink>
													)}
												</ElTooltip>
											),
										}}
									</ElTableColumn>
									<ElTableColumn
										label="大小"
										prop="name"
										align="center"
										width={180}>
										{{
											default: (scope: { row: Item & { loading?: boolean } }) =>
												scope.row.isFile ? (
													byte2Unit(scope.row.size, 'b', 2)
												) : (
													<ElLink onClick={() => onClickClcFolder(scope.row)}>
														<ElButton text loading={scope.row.loading}>
															{typeof scope.row.size == 'number'
																? byte2Unit(scope.row.size, 'b', 2)
																: '计算'}
														</ElButton>
													</ElLink>
												),
										}}
									</ElTableColumn>

									<ElTableColumn
										prop="updateTime"
										label="修改时间"
										width={160}
										show-overflow-tooltip={true}
									/>
									<ElTableColumn
										label="操作"
										width={260}
										align="center"
										fixed="right">
										{{
											default: (scope: tableScope<Item>) => (
												<>
													<Download
														authon
														showPercentage
														chunkSize={2 * 1024 * 1024}
														url={getResourceUrl(
															scope.row.path,
															scope.row.isFile
														)}
														v-permiss={getPermissKey(
															routeHook.ViewFeature.download
														)}
													/>
													<ElButton
														text={true}
														icon={Delete}
														onClick={() => onPreview(scope.$index, scope.row)}
														v-permiss={getPermissKey(
															routeHook.ViewFeature.view
														)}
														v-slots={{ default: () => '预览' }}
													/>
													<ElButton
														text={true}
														icon={Delete}
														onClick={() => onDelete(scope.$index, scope.row)}
														v-permiss={getPermissKey(routeHook.ViewFeature.del)}
														v-slots={{ default: () => '删除' }}
													/>
												</>
											),
										}}
									</ElTableColumn>
								</>
							),
							footer: () => (
								<>
									<Diaglog
										style="width:50%;height:70%;"
										ref={diaglogRef}
										onSaved={onRefesh}
									/>
								</>
							),
						}}
					</TableView>
				</div>
			);
		};
		return rander;
	}, //end setup
});
