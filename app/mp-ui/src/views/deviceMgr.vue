<template>
	<!--监控管理-->
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						v-model="table_module.query.name"
						placeholder="设备名称"
						class="handle-input mr10"></el-input>

					<el-link
						type="primary"
						v-show="false"
						title="更多"
						@click="table_module.moreOption = !table_module.moreOption">
						<ElIcon :size="20">
							<MoreFilled />
						</ElIcon>
					</el-link>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>搜索</el-button
					>
					<el-button type="primary" :icon="Plus" @click="onOpenDialog()"
						>新增</el-button
					>
				</div>
				<el-row :gutter="24" v-if="table_module.moreOption">
					<div class="handle-box">
						<div class="el-select mr10">
							<el-link type="info" @click="setDatetime(0, 0.5)">0.5h内</el-link>
							<el-link type="info" @click="setDatetime(0, 1)">1h内</el-link>
							<el-link type="info" @click="setDatetime(1, 24)">今天</el-link>
							<el-date-picker
								style="margin-top: 3px"
								v-model="table_module.query.datetimes"
								format="YYYY-MM-DD HH:mm:ss"
								value-format="YYYY-MM-DD HH:mm:ss"
								type="datetimerange"
								range-separator="至"
								start-placeholder="开始时间"
								end-placeholder="结束时间"
								title="告警事件" />
						</div>
					</div>
				</el-row>
			</el-header>
			<el-main>
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
							prop="name"
							label="名称"
							width="90"
							align="center"
							sortable
							:show-overflow-tooltip="true"></el-table-column>

						<el-table-column
							prop="innerIp"
							label="网络地址"
							width="120"
							align="center"
							sortable
							:show-overflow-tooltip="true"></el-table-column>

						<el-table-column
							prop="sip"
							label="sip地址"
							width="190"
							align="center"
							sortable
							:show-overflow-tooltip="true"></el-table-column>

						<el-table-column
							width="160"
							prop="createTime"
							label="创建时间"
							sortable
							:show-overflow-tooltip="true"></el-table-column>
						<el-table-column label="操作" width="316" align="center">
							<template #default="scope">
								<el-button
									text
									:icon="Edit"
									@click="onOpenDialog(scope.row)"
									v-if="scope.row.deviceType != '1'">
									修改
								</el-button>
								<el-button text :icon="Delete" @click="onDelete(scope.row)">
									删除
								</el-button>
							</template>
						</el-table-column>
					</el-table>
				</el-scrollbar>
			</el-main>
			<el-footer>
				<div class="pagination">
					<el-pagination
						background
						layout="prev, pager, next,total,jumper"
						:current-page="table_module.query.pageIndex"
						:page-sizes="[100, 200, 300, 400]"
						:page-size="table_module.query.pageSize"
						:total="table_module.pageTotal"
						@current-change="onPageChange">
					</el-pagination>
				</div>
			</el-footer>
		</el-container>

		<!-- 弹出框 
		<edit-ip-camera :title="编辑" :label-width="100" ref="editIpCameraRef" @saved="onIpcamerSave()"></edit-ip-camera> 
		-->
		 
	</div>
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
		type FormRules,
		type FormInstance,
		ElTreeSelect,
		dayjs,
		ElTable
	} from 'element-plus';
	import {type TreeNode } from 'element-plus/es/components/tree-v2/src/types';
	import { type TreeNodeData } from 'element-plus/es/components/tree/src/tree.type';
	import {
		Delete,
		Edit,
		Search,
		Compass,
		MoreFilled,
		Download,
		Plus,
		Minus,
	} from '@element-plus/icons-vue';
	import * as api from '../api/device';
	import * as res_api from '../api';
	import * as site_api from '../api/site';
	import * as t from '../store/types/devices';
	import { detailsInfo } from '../components/details';
	import { imgVideo, types } from '../components/player';
	import { str2Obj, createStateEndDatetime } from '../utils';
	import { showLoading, closeLoading } from '../components/Logining';
	import { getTableIndex } from '../utils/tables';
	import   editIpCamera  from '../components/biz/src/editCamera';

	interface TableRow {
		id: number;
		uuid: string;
		deviceType: number;
		innerIp: string;
		ip: string;
		name: string;
		createTime: string;
		poster?: string;
		streams?: string;
	}
	interface Query extends IpageParam {
		name: string;
		category?: number;
		datetimes: Array<string>;
	}
	interface table_module {
		query: Query;
		moreOption: boolean;
		data: TableRow[];
		currentRow?: TableRow;
		pageTotal: number;
	}

	const tableInstance=ref<InstanceType< typeof ElTable>>(); 
	const currentTableItemIndex = ref<number>();
	const table_module = reactive<table_module>({
		query: {
			name: '',
			datetimes: [],
			pageIndex: 1,
			pageSize: 10,
			order: 'asc',
			orderBy: '',
		},
		moreOption: false,
		data: [],
		pageTotal: -1,
	});
	const setDatetime = (t: number, i: number) => {
		table_module.query.datetimes = createStateEndDatetime(t, i);
	};

	interface SiteCategory {
		List: Array<{ id: number; name: string }>;
	}

	const SiteCategoryRef = ref<SiteCategory>({ List: [] });
	const getSiteType = async () => {
		const res = await site_api.select_svc();
		if (res.code == 0) {
			SiteCategoryRef.value.List = res.data;
		}
	};
	getSiteType();
	onMounted(() => {});
	// 排序
	const onColChange = (column: any) => {
		table_module.query.order = column.order === 'descending' ? 'desc' : 'asc';
		table_module.query.orderBy = column.prop;
		if (column) getData(); // 获取数据的方法
	};
	const tableRowProp = (data: { row: any; rowIndex: number }) => {
		data.row.index = data.rowIndex;
		return ''
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
		getQuery(),
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
	getData();
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
		if (tableInstance.value&&
			tableInstance.value.data &&
			index > -1 &&
			index < tableInstance.value.data.length
		) {
			let row = tableInstance.value.data[index];
			tableInstance.value.setCurrentRow(row);
			onTableSelect(row);
		}
	};
	const onTableSelect = (row: any) => {
		currentTableItemIndex.value = row.index;
		table_module.currentRow = row;
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
				if (tableInstance.value&&0 <= v && v < tableInstance.value.data.length) {
					setTableSelectItem(v);
				} else {
					if (v < 0) ElMessage.error('已经是第一条了');
					else if (tableInstance.value&&v >= tableInstance.value.data.length)
						ElMessage.error('已经是最后一条了');
				}
			}
		}
		//process_view.value.keyDown(e)
		e.stopPropagation();
	};

	// 删除操作
	const onDelete = (row: any) => {
		// 二次确认删除
		ElMessageBox.confirm(`确定要删除"${row.name}"任务吗？`, '提示', {
			type: 'warning',
		})
			.then(() => {
				api
					.del_camera_svc(row.id)
					.then((res) => {
						if (res.code == 0) ElMessage.success('删除成功'), getData();
						else ElMessage.error(`删除失败:${res.message}`);
					})
					.finally(() => {});
			})
			.catch(() => {});
	};
	//**监控球机 */
	const editIpCameraRef = ref<InstanceType<typeof editIpCamera>>();
	const onOpenDialog = (row?: any) => { 
		//有记录编辑无数据增加  
		//editIpCameraRef.value.openDialog(row.siteId, row.id);
	}; 
	const onIpcamerSave = () => {
		getData();
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

	::v-deep .streamInfo {
		max-width: 560px;

		.el-card {
			margin: 2px 0;
		}

		.el-form-item {
			padding: 8px 0;
		}

		.el-form-item__content {
			width: 470px;
		}

		.el-card__body {
			padding: 2px 5px;
		}

		.el-form-item__label {
			width: 70px;
		}
	}
</style>
