<template>
	<el-card class="box-card">
		<!--header-->
		<template #header>
			<div class="card-header">
				<el-input v-model="tree_module.query.name" placeholder="点位名称">
					<template #append>
						<el-button :icon="Search" @click="tree_module.onSearch" />
					</template>
				</el-input>
			</div>
		</template>
		<!--content-->
		<el-scrollbar>
			<div class="content">
				<el-tree
					v-if="hasData"
					highlight-current
					@node-click="onNodeCheck"
					ref="treeRef"
					class="filter-tree"
					:data="tree_module.data"
					:props="tree_module.defaultProps"
					default-expand-all
					:filter-node-method="tree_module.filterNode">
					<template #default="{ node, data }">
						<span>
							<!-- 没有子级所展示的图标 -->
							<i v-if="data.devices"
								><el-icon> <Avatar /> </el-icon
							></i>
							<i v-else>
								<el-icon
									:class="[
										{ 'is-loading': data.state == 0 },
										'state_' + data.state,
									]">
									<component :is="data.statueComponent" /> </el-icon
							></i>
							<span class="label">
								<el-tooltip :content="data.deviceDesc || node.label">
									<el-text truncated>{{ node.label }} </el-text>
								</el-tooltip>
							</span>
						</span>
					</template>
				</el-tree>
				<el-empty v-else></el-empty>
			</div>
		</el-scrollbar>
		<!--footer-->
		<template #footer>
			<div class="context">
				<el-pagination
					v-if="hasData"
					background
					layout="prev,next"
					:total="tree_module.total"
					:current-page="tree_module.query.pageIndex"
					:page-size="tree_module.query.pageSize"
					@current-change="tree_module.pageChange" />
			</div>
		</template>
	</el-card>
</template>
<script setup lang="ts">
	import { ref, reactive, computed, onMounted, onUnmounted } from 'vue';
	import {
		ElMessage,
		ElMessageBox,
		FormRules,
		FormInstance,
		ElTreeSelect,
		ElTree,
	} from 'element-plus';
	import {
		TreeNode,
		TreeOptionProps,
	} from 'element-plus/es/components/tree-v2/src/types';
	import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type';
	import {
		Delete,
		Edit,
		Search,
		Compass,
		MoreFilled,
		Download,
		CloseBold,
		VideoCamera,
		Avatar,
		ArrowUp,
		ArrowDown,
		Loading,
	} from '@element-plus/icons-vue';
	import * as api from '../../..//api/site';
	import * as gb_api from '../../..//api/deviceState';
	import { showLoading, closeLoading } from '../../../components/Logining';
	import * as types from './types';

	interface Emits {
		(e: 'nodeClick', streams: types.Stream, device: types.DeviceData): void;
	}
	const emits = defineEmits<Emits>();
	const treeRef = ref<InstanceType<typeof ElTree>>();
	interface Tree {
		[key: string]: any;
	}
	interface Query extends IpageParam {
		name: string;
	}

	interface tree_module {
		query: Query;
		data: Array<types.Site>;
		currentItem?: types.Site;
		currentDevice?: types.DeviceData;
		total: number;
		defaultProps: TreeOptionProps; // { children: String; label:(treeData:any,treeNode:any)=>string };
		filterNode: (value: string, data: Tree) => boolean;
		pageChange: (val: number) => void;
		onSearch: () => void;
	}
	const tree_module = reactive<tree_module>({
		query: {
			name: '',
			pageIndex: 1,
			pageSize: 20,
			order: 'asc',
			orderBy: '',
		},
		data: [],
		total: 0,
		filterNode: (value: string, data: Tree) => {
			if (!value) return true;
			return data.label.includes(value);
		},
		// 分页导航
		pageChange: (val: number) => {
			tree_module.query.pageIndex = val;
			getData();
		},
		onSearch: () => {
			getData();
		},
		defaultProps: {
			children: 'devices',
			label: 'name',
		},
	});
	// 获取表格数据
	const getData = () => {
		showLoading();
		api
			.list_svc(tree_module.query)
			.then((res) => {
				if (res.code == 0) {
					for (let i = 0; i < res.data.length; i++) {
						let devictState = types.DeviceState.loading;
						//1. 如果 监控球机 0，删除
						if (res.data[i].devices && res.data[i].devices.length == 0) {
							setStatueComponent(res.data[i], devictState);
							delete res.data[i].devices;
						} //2. 如果 监控球机 1，移动值 为 res.data[i] 属性
						else if (res.data[i].devices && res.data[i].devices.length == 1) {
							res.data[i].device = res.data[i].devices[0];
							setStatueComponent(res.data[i], devictState);
							delete res.data[i].devices;
						} else {
							//3. 多个监控球机 :多
							for (let j = 0; j < res.data[i].devices.length; j++) {
								setStatueComponent(res.data[i].devices[j], devictState);
							}
						}
					}
					tree_module.data = res.data;
					tree_module.total = res.total || -1;
				} else {
					ElMessage.error(res.message);
				}
			})
			.finally(() => {
				closeLoading();
			});
	};
	const hasData = computed(() => tree_module.data.length > 0);

	const onNodeCheck = (row: types.Site | types.DeviceData) => {
		let data = row as types.Site;
		let device = row as types.DeviceData;
		if (data) {
			tree_module.currentItem = data;
			//只有一个设备的点
			if (data.device) {
				let device = data.device;
				tree_module.currentDevice = device;
				let stream = device.streams;
				emits('nodeClick', stream, device);
			}
			//有多个设备的点 ，仅展开
			else if (data.devices) console.info('展开');
		}
		if (device) {
			//点位
			tree_module.currentDevice = device;
			emits('nodeClick', device.streams, device);
		}
	};
	// 状态
	const state_0 = Loading;
	const state_1 = VideoCamera;

	const setStatueComponent = (
		data: types.Site | types.DeviceData,
		state: types.DeviceState
	) => {
		data.state = state;
		switch (state) {
			case types.DeviceState.loading:
				data.statueComponent = state_0;
				break;
			case types.DeviceState.Connected:
				data.statueComponent = state_1;
				break;
			default:
				data.statueComponent = state_1;
				break;
		}
	};

	let timer: NodeJS.Timeout | null = null;
	const onSyncState = () => {
		if (timer) clearInterval(timer);
		gb_api
			.get_gb_device_state()
			.then((res) => { 
        let stateArray=res.data
        console.info("gb device State:",stateArray)
				for (let i = 0; i < tree_module.data.length; i++) {
					let site = tree_module.data[i];
					if (site.box) {
						if (site.devices) {
							for (let j = 0; j < site.devices.length; j++) {
								setStatueComponent(
									site.devices[j],
									types.DeviceState.Connected
								);
							}
						} else if (site.device) {
							setStatueComponent(site, types.DeviceState.Disconected);
						}
					} else {
						if (site.devices) {
							for (let j = 0; j < site.devices.length; j++) {
								setStatueComponent(
									site.devices[j],
									types.DeviceState.Disconected
								);
							}
						} else if (site.device) {
							setStatueComponent(site, types.DeviceState.Disconected);
						}
						setStatueComponent(site, types.DeviceState.Disconected);
					}
				}
			})
			.finally(() => {
				timer = setInterval(onSyncState, 30000);
			});

      gb_api
			.get_rtc_device_state().then((res)=>{ console.info("rtc",res) })
	};
	onMounted(() => {
		getData();
		timer = setInterval(() => {
			onSyncState();
		}, 500);
	});
	onUnmounted(() => {
		if (timer) clearInterval(timer);
		timer = null;
	});
</script>
<style lang="less" scoped>
	::v-deep .el-card__body {
		padding-top: 5px;
		height: 100%;
		.content {
			padding: 0;
			.el-tree-node__content {
				margin-left: -24px;
			}
			.label {
				overflow: hidden;
				text-overflow: ellipsis;
				white-space: nowrap;
				padding: 0 5px;
			}
		}
	}
	.state_0 {
		color: #b3b02f;
	}
	.state_1 {
		color: green;
	}
	.state_2 {
		color: red;
	}
</style>
