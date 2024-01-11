<template>
	<video
		ref="video"
		v-if="option.url"
		:src="option.url"
		controls
		autoplay
		:poster="option.poster"></video>
	<el-empty v-else description="未加载数据" />
</template>

<script lang="ts" setup>
	import {
		watch,
		watchEffect,
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
	const video = ref();
	//ios
	watch(
		() => props.option.url,
		(u, o) => {
			if (u) {
				//console.info("ios")
				video.value.load();
				video.value.play();
			}
		}
	);
</script>
<style scoped lang="less">
	video {
		width: 100%;
		height: 100%;
		object-fit: fill;
	}
</style>
