<template >   
	<el-image v-if="option.url"  v-loading="loading"
		:src="result"
		style="width: 100%; height: 100%"
		:title="option.name"
		:zoom-rate="1.2"
		:max-scale="7"
		:min-scale="0.2"
		fit="cover"
		:preview-src-list="srcList"
	></el-image>
	<el-empty v-else description="未加载数据" />
</template>

<script lang="ts" setup>
	import { watch, type PropType, ref, computed } from 'vue';
	import {type imageOption } from './types';
	import * as res_api from '../../../api';
	const props = defineProps({
		option: {
			type: Object as PropType<imageOption>,
			required: true,
		},
	});
	const result = ref('');
	const loading=ref(true)
	const srcList = computed(() => [props.option.url]);
	watch(
		() => props.option,
		(n, o) => {
			loading.value=true;
			res_api
				.request_resource_svc(props.option.url)
				.then((response) => (result.value = response))
				.finally(()=>loading.value=false)
		},
		{ immediate: true }
	);
</script>
