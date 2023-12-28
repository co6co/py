<template>
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-select
						style="width: 160px"
						class="mr10"
						clearable
						v-model="table_module.query.alarmType"
						placeholder="告警类型">
						<el-option
							v-for="(item, index) in table_module.categoryList"
							:key="index"
							:label="item.desc"
							:value="item.alarmType" />
					</el-select>
					<div class="el-select mr10">
						<el-date-picker
							style="margin-top: 3px"
							v-model="table_module.query.datetimes"
							format="YYYY-MM-DD HH:mm:ss"
							value-format="YYYY-MM-DD HH:mm:ss"
							type="datetimerange"
							range-separator="至"
							start-placeholder="开始时间"
							end-placeholder="结束时间"
							title="告警时间" />
						<el-link type="info" @click="setDatetime(0, 0.5)">30分钟内</el-link>
						<el-link type="info" @click="setDatetime(0, 1)">1小时内</el-link>
						<el-link type="info" @click="setDatetime(1, 24)">今天</el-link>
					</div>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>搜索</el-button
					>
					<el-button
						type="info"
						v-show="false"
						@click="table_module.moreOption = !table_module.moreOption"
						>更多<el-icon :is="moreIcon"
							><component :is="moreIcon"></component> </el-icon
					></el-button>
				</div>
				<el-row :gutter="24" v-if="table_module.moreOption">
					<div class="handle-box">
						<!--更多信息-->
					</div>
				</el-row>
			</el-header>

			<el-main>
				<el-scrollbar>
					<!--主内容-->
					<el-row>
						<el-col :span="12">
							<div style="height: 100%; overflow: auto">
								<el-table
									highlight-current-row
									@sort-change="onColChange"
									:row-class-name="tableRowProp"
									:data="table_module.data"
									border
									class="table"
									ref="tableInstance"
									@row-click="onTableSelect"
									header-cell-class-name="table-header">
									<el-table-column
										label="序号"
										width="119"
										align="center"
										:show-overflow-tooltip="true">
										<template #default="scope">
											<span>{{
												getTableIndex(table_module.query, scope.$index)
											}}</span>
										</template>
									</el-table-column>
									<el-table-column
										prop="alarmTypePO.desc"
										label="安全员"
										width="119"
										sortable
										:show-overflow-tooltip="true"></el-table-column>

									<el-table-column
										prop="alarmTypePO.desc"
										label="告警类型"
										width="119"
										sortable
										:show-overflow-tooltip="true"></el-table-column>

									<el-table-column
										width="160"
										prop="alarmTime"
										label="告警时间"
										sortable
										:show-overflow-tooltip="true"></el-table-column>
								</el-table>
							</div>
						</el-col>
						<el-col :span="12" style="position: relative;">
							<div style="padding: 5px; overflow: hidden; ">
								<img-video    :viewOption="form2.data"></img-video>
							</div>
						</el-col>
					</el-row>
				</el-scrollbar>
			</el-main>

			<el-footer>
				<!--分页组件-->
				<div class="pagination">
					<el-pagination
						background
						layout="prev, pager,next,total,jumper"
						:current-page="table_module.query.pageIndex"
						:page-sizes="[100, 200, 300, 400]"
						:page-size="table_module.query.pageSize"
						:total="table_module.pageTotal"
						@current-change="onPageChange">
					</el-pagination>
				</div>
			</el-footer>
		</el-container>

		<!-- 弹出框 -->
		<el-dialog
			title="详细信息"
			v-model="form.dialogVisible"
			style="width: 80%; height: 76%"
			@keydown.ctrl="keyDown">
			<details-info :data="form.data"></details-info>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="form.dialogVisible = false">取 消</el-button>
				</span>
			</template>
		</el-dialog>

		<!-- 弹出框 -->
		<el-dialog
			title="详细信息"
			v-model="form2.dialogVisible"
			style="width: 98%; height: 90%"
			@keydown.ctrl="keyDown">
			<el-row>
				<el-col :span="12">
					<img-video :viewOption="form2.data"></img-video>
				</el-col>
				<el-col :span="12"> </el-col>
			</el-row>
			<template #footer>
				<span class="dialog-footer">
					<el-button @click="form2.dialogVisible = false">取 消</el-button>
				</span>
			</template>
		</el-dialog>
	</div>
</template>

<script setup lang="ts" name="basetable">
	import {
		ref,
		watch,
		reactive,
		nextTick,
		PropType,
		onMounted,
		onBeforeUnmount,
		computed,
	} from 'vue';
	import {
		ElMessage,
		ElMessageBox,
		FormRules,
		FormInstance,
		ElTreeSelect,
		dayjs,
	} from 'element-plus';
	import { TreeNode } from 'element-plus/es/components/tree-v2/src/types';
	import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type';
	import {
		Delete,
		Edit,
		Search,
		Compass,
		MoreFilled,
		Download,
		ArrowUp,
		ArrowDown,
	} from '@element-plus/icons-vue';
	import * as api from '../api/alarm';
	import * as res_api from '../api';
	import { detailsInfo } from '../components/details';
	import { imgVideo, types } from '../components/player';
	import { str2Obj, createStateEndDatetime } from '../utils';

	import { showLoading, closeLoading } from '../components/Logining';
	import { getTableIndex } from '../utils/tables';

	interface TableRow {
		id: number;
		uuid: string;
		alarmType: string;
		videoUid: string;
		rawImageUid: string;
		markedImageUid: string;
		alarmTime: string;
		createTime: string;
	}
	interface AlertCategory {
		alarmType: string;
		desc: string;
	}
	interface Query extends IpageParam {
		datetimes: Array<string>;
		alarmType: String;
	}
	interface table_module {
		query: Query;
		moreOption: boolean;
		data: TableRow[];
		currentRow?: TableRow;
		pageTotal: number;
		categoryList: AlertCategory[];
	}
	const elTreeInstance = ref<any>(null);
	const selectedVal = ref<any[]>();
	const cacheData = [{ value: 5, label: '位置信息' }];
	const form_attach_data = {};
	const tableInstance = ref<any>(null);
	const currentTableItemIndex = ref<number>();
	const table_module = reactive<table_module>({
		query: {
			alarmType: '',
			datetimes: [],
			pageIndex: 1,
			pageSize: 15,
			order: 'asc',
			orderBy: '',
		},
		moreOption: false,
		data: [],
		pageTotal: -1,
		categoryList: [],
	});
	const moreIcon = computed(() => {
		if (table_module.moreOption) return ArrowUp;
		else return ArrowDown;
	});
	const setDatetime = (t: number, i: number) => {
		table_module.query.datetimes = createStateEndDatetime(t, i);
	};
	// 排序
	const onColChange = (column: any) => {
		table_module.query.order = column.order === 'descending' ? 'desc' : 'asc';
		table_module.query.orderBy = column.prop;
		if (column) getData(); // 获取数据的方法
	};

	const tableRowProp = (data: { row: any; rowIndex: number }) => {
		data.row.index = data.rowIndex;
	};
	const onRefesh = () => {
		getData();
	};
	// 查询操作
	const onSearch = () => {
		getData();
	};
	const getQuery = () => {
		table_module.query.pageIndex = table_module.query.pageIndex || 1;
	};
	// 获取表格数据
	const getData = () => {
		showLoading();
		getQuery();
		api.list_svc(table_module.query).then((res) => {
			if (res.code == 0) {
				table_module.data = res.data;
				table_module.pageTotal = res.total || -1;
			} else {
				ElMessage.error(res.message);
			}
			closeLoading();
		});
	};

	const getAlarmCategory = async () => {
		const res = await api.alert_category_svc();
		if (res.code == 0) {
			table_module.categoryList = res.data;
		}
		getData();
	};
	getAlarmCategory();

	// 分页导航
	const onPageChange = (pageIndex: number) => {
		let totalPage = Math.ceil(
			table_module.pageTotal / table_module.query.pageSize.valueOf()
		);
		if (pageIndex < 1) ElMessage.error('已经是第一页了');
		else if (pageIndex > totalPage) ElMessage.error('已经是最后一页了');
		else (table_module.query.pageIndex = pageIndex), getData();
	};
	const setTableSelectItem = (index: number) => {
		if (
			tableInstance._value.data &&
			index > -1 &&
			index < tableInstance._value.data.length
		) {
			let row = tableInstance._value.data[index];
			tableInstance._value.setCurrentRow(row);
			onTableSelect(row);
		}
	};
	const onTableSelect = (row: any) => {
		currentTableItemIndex.value = row.index;
		table_module.currentRow = row;
		onOpen2Dialog(row);
	};
	const keyDown = (e: KeyboardEvent) => {
		if (e.ctrlKey) {
			if (['ArrowLeft', 'ArrowRight'].indexOf(e.key) > -1) {
				let current = table_module.query.pageIndex.valueOf();
				let v =
					e.key == 'ArrowRight' || e.key == 'd' ? current + 1 : current - 1;
				onPageChange(v);
			}
			if (['ArrowUp', 'ArrowDown'].indexOf(e.key) > -1) {
				let current = currentTableItemIndex.value;
				if (!current) current = 0;
				let v =
					e.key == 'ArrowDown' || e.key == 's' ? current + 1 : current - 1;
				if (0 <= v && v < tableInstance._value.data.length) {
					setTableSelectItem(v);
				} else {
					if (v < 0) ElMessage.error('已经是第一条了');
					else if (v >= tableInstance._value.data.length)
						ElMessage.error('已经是最后一条了');
				}
			}
		}
		//process_view.value.keyDown(e)
		e.stopPropagation();
	};
	//**详细信息 */
	interface dialogDataType {
		dialogVisible: boolean;
		data: Array<any>;
	}
	let dialogData = {
		dialogVisible: false,
		data: [],
	};
	let form = reactive<dialogDataType>(dialogData);
	const onOpenDialog = (row?: any) => {
		form.dialogVisible = true;
		table_module.currentRow = row;
		form.data = [
			{ name: '检测结果信息', data: str2Obj(row.alarmAttachPO.result) },
			{ name: '视频流信息', data: str2Obj(row.alarmAttachPO.media) },
			{ name: 'GPS信息', data: str2Obj(row.alarmAttachPO.gps) },
		];
	};
	//**查看视频图片信息 */
	interface dialog2DataType {
		dialogVisible: boolean;
		data: Array<types.resourceOption>;
	}
	let dialog2Data = {
		dialogVisible: false,
		data: [],
	};
	let form2 = ref<dialog2DataType>(dialog2Data);
	const setVideoResource = (uuid: string, option: types.videoOption) => {
		res_api
			.request_resource_svc(
				import.meta.env.VITE_BASE_URL + `/api/resource/poster/${uuid}`
			)
			.then((res) => {
				option.poster = res;
			})
			.catch((e) => (option.poster = ''));
		res_api
			.request_resource_svc(
				import.meta.env.VITE_BASE_URL + `/api/resource/${uuid}`
			)
			.then((res) => {
				option.url = res;
			})
			.catch((e) => (option.url = ''));
	};
	const setImageResource = (uuid: string, option: types.imageOption) => {
		res_api
			.request_resource_svc(
				import.meta.env.VITE_BASE_URL + `/api/resource/${uuid}`
			)
			.then((res) => {
				option.url = res;
			})
			.catch((e) => (option.url = ''));
	};
	const getResultUrl = (uuid: string, isposter: boolean = false) => {
		if (isposter)
			return (
				import.meta.env.VITE_BASE_URL + `/api/resource/poster/${uuid}/700/600`
			);
		return import.meta.env.VITE_BASE_URL + `/api/resource/${uuid}`;
	};
	const onOpen2Dialog = (row: TableRow) => {
		//form2.value.dialogVisible = true;
		//table_module.currentRow = row;
		form2.value.data = [
			{
				url: getResultUrl(row.rawImageUid),
				name: '原始图片',
				poster: getResultUrl(row.rawImageUid, true),
				type: 1,
			},
			{
				url: getResultUrl(row.markedImageUid),
				name: '标注图片',
				poster: getResultUrl(row.markedImageUid, true),
				type: 1,
			},
			{
				url: getResultUrl(row.videoUid),
				name: '原始视频',
				poster: getResultUrl(row.videoUid, true),
				type: 0,
			},
		];
	};
</script>
<style scoped lang="less">
	@import '../assets/css/tables.css';

	.view .title {
		color: var(--el-text-color-regular);
		font-size: 18px;
		margin: 10px 0;
	}
	.view .value {
		color: var(--el-text-color-primary);
		font-size: 16px;
		margin: 10px 0;
	}

	::v-deep .view .radius {
		height: 40px;
		width: 70%;
		border: 1px solid var(--el-border-color);
		border-radius: 0;
		margin-top: 20px;
	}
	::v-deep .el-table tr,
	.el-table__row {
		cursor: pointer;
	}

	.formItem {
		display: flex;
		align-items: center;
		display: inline-block;
		.label {
			display: inline-block;
			color: #aaa;
			padding: 0 5px;
		}
	}

	::v-deep .el-dialog__body {
		height: 70%;
		overflow: auto;
	}
	.menuInfo {
		.el-menu {
			width: auto;
			.el-menu-item {
				padding: 10px;
				height: 40px;
			}
		}
	}
</style>
