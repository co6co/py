<template>
	<div>
		<search
			v-model="vue_module.query.name"
			placeholder="定位名称"
			@search="onQuery" />
		<van-pull-refresh v-model="vue_module.statue.loading" @refresh="onRefresh">
			<van-list
				@load="getData"
				v-model:error="vue_module.statue.error"
				error-text="请求失败，点击重新加载"
				direction="down"
				:loading="vue_module.statue.loading"
				:finished="vue_module.statue.finished"
				finished-text="没有更多了">
				<card
					v-for="item in vue_module.data"
					:key="item.id"
					:title="item.name"
					:thumb="getPoster(item.id)"
					@click="onPreview(item)">
					{{ item }} 
					<template #tags>
						<van-tag plain type="danger">设备地址{{ item.innerIp }}</van-tag>
						<van-tag type="primary">{{ item.name }}</van-tag>
					</template>
				</card>
				<empty v-if="vue_module.statue.empty" description="无数据" />
			</van-list>
		</van-pull-refresh>
	</div>
</template>
<script setup lang="ts">
	import { ref, reactive, watchEffect } from 'vue';
	import {
		Image as VanImage,
		PullRefresh as vanPullRefresh,
		Divider,
		List as vanList,
		Cell,
		Search,
		Icon,
		Grid,
		GridItem,
		Card,
		Tag as vanTag,
		NavBar,
	} from 'vant';

	import * as api from '../api/device';
	import * as res_api from '../api';
	import * as d from '../store/types/devices';

	import { useAppDataStore } from '../store/appStore';
	import { useRouter } from 'vue-router';
	import { showNotify, Empty } from 'vant';
	import { json } from 'stream/consumers';
	const router = useRouter();
	const dataStore = useAppDataStore();

	interface Tree {
		[key: string]: any;
	}
	interface Query extends IpageParam {
		name: string;
	}

	interface vue_module {
		query: Query;
		data: Array<d.dataItem>;
		currentItem?: d.dataItem;
		total: number;
		statue: {
			loading: boolean;
			finished: boolean;
			refreshing: boolean;
			error: boolean;
			empty: boolean;
		};
	}

	const vue_module = reactive<vue_module>({
		query: {
			name: '',
			pageIndex: 1,
			pageSize: 10,
		},
		statue: {
			loading: false,
			finished: false,
			refreshing: false,
			error: false,
			empty: true,
		},
		data: [],
		total: -1,
	});
	watchEffect(() => {
		vue_module.statue.empty = vue_module.data.length == 0;
	});

	const getData = () => {
		vue_module.statue.loading = true;
		vue_module.statue.finished = false;
		api
			.list_svc(vue_module.query)
			.then((res) => {
				if (res.code == 0) {
					vue_module.data.push(...res.data);
					vue_module.total = res.total || -1;
				} else showNotify({ type: 'danger', message: res.message });
				if (vue_module.data.length >= res.total) {
					vue_module.statue.finished = true;
				}
				if (res.data.length > 0) {
					vue_module.query.pageIndex++;
				}
			})
			.catch(() => {
				vue_module.statue.error = true;
			})
			.finally(() => {
				vue_module.statue.loading = false;
			});
	};
	const onQuery = () => {
		vue_module.data = [];
		vue_module.query.pageIndex = 1;
		getData();
	};
	const onRefresh = () => {
		// 清空列表数据
		vue_module.query.pageIndex = 1;
		vue_module.data = [];
		getData();
	};
	const ddd = reactive({ data: {} });

	const onPreview = (row: d.dataItem) => {
		console.info(typeof row.streams == 'string');
		if (row.streams && typeof row.streams == 'string')
			row.streams = JSON.parse(row.streams);
		dataStore.setState(row);
		router.push({
			path: '/preview.html',
		});
	};
	const getPoster = (id: number) => {
		return import.meta.env.VITE_BASE_URL + '/api/biz/device/poster/' + id;
	};
</script>
