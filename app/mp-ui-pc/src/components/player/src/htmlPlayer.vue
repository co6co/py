<template>
    <video 	v-if="option.url" :src="option.url" controls  autoplay :poster="option.poster"></video> 
	<el-empty v-else description="未加载数据" />
</template>

<script lang="ts" setup>
	import {
		watch,
		PropType,
		reactive,
		ref,
		computed,
		onMounted,
		onBeforeUnmount,
		nextTick,
	} from 'vue'; 
	import { videoOption } from './types';
	const props = defineProps({
		option: {
			type: Object as PropType<videoOption>,
			required: true,
		},
	});  
</script>
<style scoped lang="less">
video{width: 100%;height: 100%;object-fit: fill;}
</style>