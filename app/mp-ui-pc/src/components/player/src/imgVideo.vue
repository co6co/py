<template>
	<el-row class="Image">
		<!--显示大图-->
		<el-col :span="24" >
			<component
				:key="currentName"
				:is="compoents[currentName]"
				:option="current_options"></component>
		</el-col>
	</el-row>
	<el-row class="NavImage">
		<!--导航图-->
		<el-col :span="24">
			<div class="imag_nav_container">
				<el-scrollbar>
					<ul>
						<li
							@click="onShow(item, index)"
							:key="index"
							v-for="(item, index) in imageOptions">
							<a
								><el-image :key="index" :src="item.poster" :title="item.name" />
							</a>
						</li>

						<li
							@click="onShow(item, index)"
							:key="index"
							v-for="(item, index) in videoOptions"
							style="position: relative">
							<a>
								<el-image :src="item.poster" :title="item.name" /><CaretRight />
							</a>
						</li>
					</ul>
				</el-scrollbar>
			</div>
		</el-col>
	</el-row>
</template>
<script setup lang="ts">
	import {
		watch,
		PropType,
		reactive,
		ref,
		markRaw,
		defineAsyncComponent,
		computed,
		onMounted,
		onBeforeUnmount,
		nextTick,
	} from 'vue';
	import 'vue3-video-play/dist/style.css';
	import { resourceOption } from './types';
	import { ElImage } from 'element-plus';
	const props = defineProps({
		viewOption: {
			type: Array<resourceOption>,
			required: true,
		},
	});
	const compoents = reactive({
		Image: markRaw(
			defineAsyncComponent(
				() => import('../../../components/player/src/Image.vue')
			)
		),
		Video: markRaw(
			defineAsyncComponent(
				() => import('../../../components/player/src/Player.vue')
			)
		),
		Splayer: markRaw(
			defineAsyncComponent(
				() => import('../../../components/player/src/Splayer.vue')
			)
		),
	});
	const imageOptions = computed(() => {
		return props.viewOption.filter((m) => m.type == 1);
	});

	const videoOptions = computed(() => {
		return props.viewOption.filter((m) => m.type == 0);
	});
	const currentName = ref<'Image' | 'Video' | 'Splayer'>('Image');
	const current_options = ref<resourceOption>({
		url: '',
		name: 'string',
		type: 1,
		poster: '',
	});

	const onShow = (option: resourceOption, key: number) => {
		if (option.type == 0) {
			currentName.value = 'Video';
		} else currentName.value = 'Image';
		current_options.value = option;
	};
	onMounted(() => {
		if (props.viewOption.length > 0)
			current_options.value = props.viewOption[0];
	});
</script>
<style scoped lang="less">
	.Image {
		.el-col {
			height: 25rem; //rem 相对与 html 的 font-size计算
						   //em 相对与 父元素的  font-size计算
							//1vh=  1/100浏览器高度
							//1vw
		}
	}
	.imag_nav_container {
		overflow: auto;
		white-space: nowrap;
	}
	ul li {
		width: 100%;

		cursor: pointer;
		overflow: hidden;
		z-index: calc(var(--el-index-normal) - 1);
		display: inline-block; //块级元素 超出盒子会换行
		width: 200px;
		transform: none;
		border: 1px solid #ccc;
		padding: 1px;
		a {
			.el-image {
				width: 100%;
				height: 106px;
			}
			svg {
				position: absolute;
				left: 40%;
				top: 35%;
				width: 20%;
			}
		}
	}
</style>
