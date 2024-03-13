<template> 
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

			<el-main class="ui">
				<!--主内容-->
				<el-container >
					<el-header  >
						<div style="width: 100%; padding: 0 5px; overflow: hidden">
							<img-video :viewOption="form2.data"></img-video>
						</div>
					</el-header>
					<el-main  class="tables">
						<el-scrollbar>
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
									width="90"
									align="center"
									:show-overflow-tooltip="true">
									<template #default="scope">
										<span>{{
											getTableIndex(table_module.query, scope.$index)
										}}</span>
									</template>
								</el-table-column>
								<el-table-column
									prop="siteName"
									label="安全员"
									width="80"
									sortable
									:show-overflow-tooltip="true"></el-table-column>

								<el-table-column
									prop="alarmTypeDesc"
									label="告警类型"
									width="90"
									sortable
									:show-overflow-tooltip="true"></el-table-column>

								<el-table-column
									width="160"
									prop="alarmTime"
									label="告警时间"
									sortable
									:show-overflow-tooltip="true"></el-table-column>
							</el-table>
						</el-scrollbar>
					</el-main>
				</el-container>
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
</template>

<script setup lang="ts" name="basetable">
	import {
		ref,
		watch,
		reactive,
		nextTick,
		type PropType,
		onMounted,
		onBeforeUnmount,
		computed,
	} from 'vue';
	import {
		ElMessage,
		ElMessageBox,
		type	FormRules,
		type	FormInstance,
		ElTreeSelect,
		dayjs,
	} from 'element-plus';
	import {type TreeNode } from 'element-plus/es/components/tree-v2/src/types';
	import {type TreeNodeData } from 'element-plus/es/components/tree/src/tree.type';
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
	import { type AlarmItem,getResources } from '../components/biz';

	 
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
		data: AlarmItem[];
		currentRow?: AlarmItem;
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
	 
	// 排序
	const onColChange = (column: any) => {
		table_module.query.order = column.order === 'descending' ? 'desc' : 'asc';
		table_module.query.orderBy = column.prop;
		if (column) getData(); // 获取数据的方法
	};

	const tableRowProp = (data: { row: any; rowIndex: number }) => {
		data.row.index = data.rowIndex;
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
		api
			.list_svc(table_module.query)
			.then((res) => {
				if (res.code == 0) {
					table_module.data = res.data;
					table_module.pageTotal = res.total || -1;
				} else {
					ElMessage.error(res.message);
				}
			})
			.finally(() => {
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
	 
	const onTableSelect = (row: any) => {
		currentTableItemIndex.value = row.index;
		table_module.currentRow = row;
		onOpen2Dialog(row);
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
	const onOpen2Dialog = (row: AlarmItem) => { 
		form2.value.data = getResources(row)
	};
</script>
<style scoped lang="less">
	@import '../assets/css/tables.css'; 
	.el-header{height:  5vh}
	#app .content .ui .el-container{height: 100%;}
	.el-main.ui { 
		overflow: auto;
		height:   87vh;
		padding: 8px 0;
		.el-header,
		.el-main {
			padding: 0;
			margin: 0;
			height:27vh;
		}
		.el-header{height:60vh;} 
		::v-deep .Image {
			.el-col {
				height:20rem;
			}
		}
		::v-deep .NavImage{height:10rem;} 
	}
	.el-footer{height:  8vh}
 
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
