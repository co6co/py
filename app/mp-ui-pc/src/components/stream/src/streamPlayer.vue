<template><div class="jess_player" ref="jess_player_container"></div></template>

<script setup lang="ts">
	import {
		watch,
		PropType,
		reactive,
		ref,
		computed,
		onMounted,
		onUnmounted,
		onBeforeUnmount,
		nextTick,
		watchEffect,
	} from 'vue';
	import '../../../assets/jessi/jessibuca-pro-demo.js';
	import '../../../assets/jessi/jessibuca-pro-talk-demo.js';
	import '../../../assets/jessi/demo.js';

	var showOperateBtns = false; // 是否显示按钮
	const props = defineProps({
		stream: {
			type: String,
			//required: true,
		},
		option: {
			type: Object as PropType<player_option>,
			required: false,
		},
	});

	const fps = reactive({ fps: 0, dfps: 0 });
	const player_option = reactive<PlayerOption>({
		type: 'MediaSource',
		renderDom: 'video',
		useWebGPU: false,
		videoBuffer: 0.2,
		videoBufferDelay: 2,
		useCanvasRender: false,
	});
	const _loaded=ref(false)
	const jess_player_container = ref<HTMLElement>();
	const jess_player = ref();
	
	const emits = defineEmits(['created', 'destroyed']); 
	
	const destroying = () => {
		emits('destroyed');
	};
	const onPlay = () => {
		try {
			if (props.stream) jess_player.value.play(props.stream);
		} catch (e) {
			console.error(e);
		}
	};
	const stop = (bck?: Function) => {  
		if (jess_player.value) {
			jess_player.value.destroy().then(() => {
				destroying();
				if (bck) bck();
			});
		} else {
			if (bck) bck();
		}
	};
	
	
	const create = () => {
		console.info('create...');
		const jessibuca = new JessibucaPro({
			container: jess_player_container.value,
			videoBuffer: player_option.videoBuffer, // 缓存时长
			videoBufferDelay: player_option.videoBufferDelay, // 1000s
			isResize: false,
			text: 'text',
			loadingText: '加载中',
			debug: false,
			debugLevel: 'debug',
			useMSE: player_option.type == 'MediaSource', // $useMSE.checked === true,
			useSIMD: player_option.type == 'Webcodec', // $useSIMD.checked === true,
			useWCS: player_option.type == 'SIMD', //$useWCS.checked === true,

			showBandwidth: showOperateBtns, // 显示网速
			//showPerformance: showOperateBtns, // 显示性能
			operateBtns: {
				fullscreen: showOperateBtns,
				screenshot: showOperateBtns,
				play: showOperateBtns,
				audio: showOperateBtns,
				//ptz: showOperateBtns,
				quality: showOperateBtns,
				performance: showOperateBtns,
			},

			timeout: 10000,
			heartTimeoutReplayUseLastFrameShow: true,
			audioEngine: 'worklet',
			forceNoOffscreen: true,
			isNotMute: false,
			heartTimeout: 10,
			ptzZoomShow: true,
			useCanvasRender: player_option.renderDom == 'canvas',
			useWebGPU: player_option.useWebGPU,
			//controlHtml:'<div>我是 <span style="color: red">test</span>文案</div>',
			supportHls265: true,
		});
		jessibuca.on('ptz', (arrow: number) => {
			console.log('ptz', arrow);
		});

		jessibuca.on('streamQualityChange', (value: string) => {
			//todo 需要调试功能
			//console.log('streamQualityChange', value,player_option.currentSource,"all=>",props.sources);
			onPlay();
		});

		jessibuca.on('timeUpdate', (value: string) => {});
		jessibuca.on('stats', (stats: { fps: number; dfps: number }) => {
			fps.dfps = stats.dfps;
			fps.fps = stats.fps;
		});
		jess_player.value = jessibuca;
		emits('created', jessibuca);
	};
	const replay = () => {
		try {
			stop(() => {
				create(), onPlay();
			});
		} catch (e) {
			console.error(e);
		}
	}; 
	watchEffect(() => {
		try {  
			if (props.stream) {
				console.error('开始播放:', props.stream);
				nextTick(() => {
					replay();
				}); // 可能会死
			} else { 
				stop();
			}
		} catch (e) {
			console.error(e);
		}
	});
	onMounted(() => { 
	});
	onUnmounted(() => {
		if (jess_player.value) jess_player.value.destroy().then(() => destroying());
	});
	const onReplay = () => {
		replay();
	};
	const onPause = () => {
		jess_player.value.pause();
	};

	defineExpose({
		stop,
	});
</script>
