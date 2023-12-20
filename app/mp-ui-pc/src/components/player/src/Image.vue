<template>
	<el-image
		:src="result"
		style="width: 100%; height: 100%"
		:title="option.name"
		:zoom-rate="1.2"
		:max-scale="7"
		:min-scale="0.2"
		fit="cover"
		:preview-src-list="srcList"
	></el-image>
</template>

<script lang="ts" setup>
	import { watch, PropType, ref, computed } from 'vue';
	import { imageOption } from './types';
	import * as res_api from '../../../api';
	const props = defineProps({
		option: {
			type: Object as PropType<imageOption>,
			required: true,
		},
	});
	const result = ref('');
	const srcList = computed(() => [props.option.url]);
	watch(
		() => props.option,
		(n, o) => {
			res_api
				.request_resource_svc(props.option.url)
				.then((response) => (result.value = response));
		},
		{ immediate: true }
	);
</script>
