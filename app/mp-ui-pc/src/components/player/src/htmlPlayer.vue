<template>
	<video
		ref="video"
		v-if="option.url"
		:src="option.url"
		controls
		autoplay
		muted
		:poster="option.poster"></video>
	<el-empty v-else description="未加载数据" />
</template>

<script lang="ts" setup>
	import { AnyNsRecord } from 'dns';
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
	const video = ref<HTMLMediaElement>();
	const check = (promise: Promise<any>) => {
		if (promise !== undefined) {
			promise
				.then((_: any) => {
					console.info('auto player.');
				})
				.catch((error: any) => {
					// Auto-play was prevented
					// Show paused UI.
					console.error(error);
				})
				.finally(() => {
					console.log('finally.', promise);
				});
		}
	};
	const fetchVideoAndPlay = () => {
		fetch(props.option.url)
			.then((response) => {
				return response.blob();
			})
			.then((blob) => {
				if (!video.value) return;
				//let ele=document.querySelector("video")
				//if (ele && 'srcObject' in ele)  {console.warn("ele存在")}
				//if (video.value && 'srcObject' in video.value)  {console.warn("view存在")}
				var mediaSource = new MediaSource();
				video.value.src = URL.createObjectURL(mediaSource);
				mediaSource.addEventListener('sourceopen', () => {
					var mime = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"';
					var sourceBuffer = mediaSource.addSourceBuffer(mime);
					blob.arrayBuffer().then((r) => {
						console.info(r);
						sourceBuffer.appendBuffer(r);
					});
				});
				//video.value.srcObject = blob;
				return video.value.play();
			})
			.then(() => {
				console.info('auto player.');
			})
			.catch((e) => {
				console.info('5 then', e);
				console.warn(video.value);
			});
	};

	const onPlay = () => {
		//fetchVideoAndPlay();
		if (!video.value) return;
		video.value.oncanplay = () => {
			if (video.value) return video.value.play();
		};
		var mediaSource = new MediaSource();
		var obj_url = URL.createObjectURL(mediaSource);
		video.value.src = obj_url;

		mediaSource.addEventListener('sourceopen', () => {
			//video/mp4; codecs="avc1.420029"; profiles="isom,iso2,avc1,mp41"
			//`video/mp4; codecs="avc1.42E01E, mp4a.40.2"`

			var sourceBuffer = mediaSource.addSourceBuffer(
				'video/mp4; codecs="avc1.420029"; profiles="isom,iso2,avc1,mp41"'
			);

			fetch(props.option.url).then(async (resp: any) => {
				var reader = resp.body.getReader();

				sourceBuffer.onupdateend = async (ev: Event) => {
					//console.log(mediaSource.readyState); // open
					const { done, value } = await reader.read();
					if (done) {
						return;
					}
					sourceBuffer.appendBuffer(value);
				};
				//sourceBuffer.onupdateend();
			});
		});
	};
	//ios

	watch(
		() => props.option.url,
		(u, o) => {
			if (u) {
				if (!video.value) return;
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
