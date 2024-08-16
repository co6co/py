import { ref, defineComponent, watch, PropType } from 'vue';
import { videoOption } from './type';
import { loadAsyncResource } from '@/api/download';
import { ElEmpty } from 'element-plus';

const props = {
	option: {
		type: Object as PropType<videoOption>,
		required: true,
	},
} as const;
export default defineComponent({
	name: 'VideoView',
	props: props,
	setup(prop, _) {
		const DATA = ref<videoOption>({
			url: '',
			poster: '',
			name: '',
		});
		const videoRef = ref<HTMLMediaElement>();

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
			fetch(prop.option.url)
				.then((response) => {
					return response.blob();
				})
				.then((blob) => {
					if (!videoRef.value) return;
					//let ele=document.querySelector("videoRef")
					//if (ele && 'srcObject' in ele)  {console.warn("ele存在")}
					//if (videoRef.value && 'srcObject' in videoRef.value)  {console.warn("view存在")}
					var mediaSource = new MediaSource();
					videoRef.value.src = URL.createObjectURL(mediaSource);
					mediaSource.addEventListener('sourceopen', () => {
						var mime = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"';
						var sourceBuffer = mediaSource.addSourceBuffer(mime);
						blob.arrayBuffer().then((r) => {
							console.info(r);
							sourceBuffer.appendBuffer(r);
						});
					});
					//videoRef.value.srcObject = blob;
					return videoRef.value.play();
				})
				.then(() => {
					console.info('auto player.');
				})
				.catch((e) => {
					console.info('5 then', e);
					console.warn(videoRef.value);
				});
		};

		const onPlay = () => {
			//fetchVideoAndPlay();
			if (!videoRef.value) return;
			videoRef.value.oncanplay = () => {
				if (videoRef.value) return videoRef.value.play();
			};
			var mediaSource = new MediaSource();
			var obj_url = URL.createObjectURL(mediaSource);
			videoRef.value.src = obj_url;
			mediaSource.addEventListener('sourceopen', () => {
				//video/mp4; codecs="avc1.420029"; profiles="isom,iso2,avc1,mp41"
				//`video/mp4; codecs="avc1.42E01E, mp4a.40.2"`
				var sourceBuffer = mediaSource.addSourceBuffer(
					'video/mp4; codecs="avc1.420029"; profiles="isom,iso2,avc1,mp41"'
				);
				fetch(prop.option.url).then(async (resp: any) => {
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

		const update = () => {
			const n = prop.option;
			if (n && n.authon) {
				if (n.url) {
					loadAsyncResource(n.url).then((res) => {
						DATA.value.url = res;
					});
				}
				if (n.poster) {
					loadAsyncResource(n.poster).then((res) => {
						DATA.value.poster = res;
					});
				}
			} else if (n) {
				if (n.url) {
					DATA.value.url = n.url;
				}
				if (n.poster) {
					DATA.value.poster = n.poster;
				}
			}
		};
		//ios
		watch(
			() => prop.option.url,
			(u, o) => {
				if (u) {
					update();
					if (!videoRef.value) return;
					videoRef.value.load();
					let promise: Promise<any> = videoRef.value.play();
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
		return () => {
			//可以写某些代码
			return (
				<>
					{prop.option.url ? (
						<video
							style="width: 100%;height: 100%;object-fit: fill;"
							ref={videoRef}
							src={DATA.value.url}
							controls={true}
							autoplay={true}
							muted
							poster={DATA.value.poster}
						/>
					) : (
						<ElEmpty v-else description="未加载数据" />
					)}
				</>
			);
		};
	},
});
