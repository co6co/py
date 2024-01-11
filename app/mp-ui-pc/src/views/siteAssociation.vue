<template>
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="handle-box">
					<el-input
						v-model="table_module.query.name"
						placeholder="站点名称"
						class="handle-input mr10"></el-input>
					<el-button type="primary" :icon="Search" @click="onSearch"
						>搜索</el-button
					>
				</div>
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
							prop="id"
							label="ID"
							width="80"
							align="center"
							sortable
							:show-overflow-tooltip="true"></el-table-column>
						<el-table-column
							prop="name"
							label="名称"
							width="120"
							align="center"
							sortable
							:show-overflow-tooltip="true"></el-table-column>
						<el-table-column
							prop="deviceCode"
							label="设备代码"
							width="120"
							align="center"
							sortable
							:show-overflow-tooltip="true"></el-table-column>

						<el-table-column label="路由器" align="center">
							<template #default="scope">
								<el-button
									text
									:icon="Message"
									@click="onOpen2Dialog('router', scope.row)">
									路由器
								</el-button>
							</template>
						</el-table-column>

						<el-table-column label="AI盒子" align="center">
							<template #default="scope">
								<el-button
									text
									:icon="Message"
									@click="onOpen2Dialog('box', scope.row)">
									AI盒子
								</el-button>
							</template>
						</el-table-column>

						<el-table-column label="监控球机" align="center">
							<template #default="scope">
								<el-button
									text
									:icon="Message"
									@click="onOpenIpCameraDialog('ip_camera', scope.row)">
									修改
								</el-button>
								<el-button
									text
									:icon="Message"
									@click="onOpen2Dialog('ip_camera', scope.row)">
									监控球机
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
						:current-page="table_module.query.pageIndex"
						:page-sizes="[10, 20, 30, 100, 200, 300, 1000]"
						layout="prev, pager, next,total,jumper"
						@update:page-size="onPageSize"
						:page-size="table_module.query.pageSize"
						:total="table_module.pageTotal"
						prev-text="上一页"
						next-text="下一页"
						@current-change="onPageChange">
					</el-pagination>
				</div>
			</el-footer>
		</el-container>
		<!-- 编辑监控球机 -->
		<edit-ip-camera ref="editIpCameraRef" :allow-modify-site="false"   @saved="onIpcamerSave()"></edit-ip-camera>
		<!-- 弹出框 -->
		<el-dialog
			:title="form2.title"
			v-model="form2.dialogVisible"
			style="width: 70%; height: 76%">
			<el-row>
				<el-col>
					<el-scrollbar>
						<details-info :data="form2.data"></details-info>
					</el-scrollbar>
				</el-col>
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
	import { ref, reactive } from 'vue';
	import { ElMessage } from 'element-plus';
	import {
		Delete,
		Edit,
		Search,
		Compass,
		MoreFilled,
		Download,
		Plus,
		Minus,
		Message,
		Cpu,
		VideoCamera,
	} from '@element-plus/icons-vue';
	import * as api from '../api/site';
	import * as dev_api from '../api/device';
	import * as res_api from '../api';
	import * as t from '../store/types/devices';
	import { detailsInfo } from '../components/details';
	import { imgVideo, types } from '../components/player';
	import { str2Obj, createStateEndDatetime } from '../utils';
	import { showLoading, closeLoading } from '../components/Logining';
	import { pagedOption, PagedOption } from '../components/tableEx';

	import { editIpCamera } from '../components/biz';

	interface TableRow {
		id: number;
		name: string;
		postionInfo: string;
		deviceCode: string;
		deviceDesc: string;
		createTime: string;
		updateTime: string;
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

	const tableInstance = ref<any>(null);
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
		getQuery(),
			api
				.list2_svc(table_module.query)
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
	const onPageSize = (size: number) => {
		table_module.query.pageSize = size;
	};
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
	};

	const queryDeviceDetailInfo = (siteId: number, deviceType: string) => {
		return api.getDetailInfo(siteId, deviceType);
	};

	//编辑球机
	const editIpCameraRef = ref(); 
	const onOpenIpCameraDialog = ( category:string, row: TableRow) => { 
    queryDeviceDetailInfo(row.id, category).then((res) => {
			if (res.code == 0) { 
				let data={siteId:row.id}
				if ( res.data.length>0)  data= res.data[0];
				if ( res.data.length>1) console.warn("site对应了多个球机,现只能编辑一个！")
				//有记录编辑无数据增加
				editIpCameraRef.value.onOpenDialog(res.data.length>0?1:0,data);
			}
		}); 
	};
  const onIpcamerSave=()=>{
    getData()
  }
	//**详细下信息 */
	interface dataContent {
		name: string;
		data: any;
	}
	interface DataType {
		title: String;
		dialogVisible: boolean;
		data: dataContent[];
	}

	let form2 = ref<DataType>({
		title: '',
		dialogVisible: false,
		data: [],
	});
	const onOpen2Dialog = (category: string, row: TableRow) => {
		form2.value.dialogVisible = true;
		if (category == 'box') {
			form2.value.title = '盒子信息';
		} else if (category == 'router') {
			form2.value.title = '路由器信息';
		} else {
			form2.value.title = '违停球信息';
		}

		queryDeviceDetailInfo(row.id, category).then((res) => {
			if (res.code == 0) {
				let data = [];
				for (let i = 0; i < res.data.length; i++) {
					data.push({ name: res.data[i].name + '信息', data: res.data[i] });
				}
				if (res.data.length == 0)
					(form2.value.dialogVisible = false),
						ElMessage.warning('未找到关联设备');
				else form2.value.data = data;
			}
		});
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
