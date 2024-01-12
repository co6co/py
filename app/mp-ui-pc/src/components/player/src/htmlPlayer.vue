<template>
	<video
		ref="video"
		v-if="option.url"
		controls
		autoplay
		muted
		:poster="option.poster">
		<source :src="option.url" type="video/mp4" />
	</video>
	<el-empty v-else description="未加载数据" />
</template>

<script lang="ts" setup>
	import { constants } from 'fs';
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

	const fetchVideoAndPlay = () => {
		fetch(props.option.url)
			.then((response) => response.blob())
			.then((blob) => {
				console.info('3 then');
				video.value.srcObject = blob;
				return video.value.play();
			})
			.then((promise) => {
				console.info('3 then', promise);
			})
			.catch((e) => {
				// Video playback failed ;(
			});
	};
	//ios
	watch(
		() => props.option.url,
		(u, o) => {
			if (u) {
				video.value.load();
				let promise: Promise<any> = video.value.play();
				if (promise !== undefined) {
					promise
						.then(function () {
							// Automatic playback started!
							console.info('auto player.');
						})
						.catch(function (error: any) {
							// Automatic playback failed.
							// Show a UI element to let the user manually start playback.
							console.error(error);
						})
						.finally(() => {
							console.log('finally.', promise);
						});
					console.log('finally..', promise);
				}
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
