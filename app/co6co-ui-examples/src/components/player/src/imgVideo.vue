<template>
	<el-row class="Image">
		<!--显示大图-->
		<el-col :span="24" > 
			<component
				:key="currentName"
				:is="components[currentName]"
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
							v-for="(item, index) in imageOptions"
							@click="onShow(item, index)"
							:key="index"
							>
							<a
								><el-image :key="index" :src="item.poster" :title="item.name" />
							</a>
						</li>

						<li 
							v-for="(item, index) in videoOptions"
							@click="onShow(item, index)"
							:key="index"
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
		reactive,
		ref,
		markRaw,
		defineAsyncComponent,
		computed, 
		nextTick,
	} from 'vue'; 
	import { type resourceOption } from './types';
	import { ElImage } from 'element-plus';
	const components = reactive({
		Image: markRaw(
			defineAsyncComponent(
				() => import('../../../components/player/src/Image.vue')
			)
		), 
		htmlPlayer: markRaw(
			defineAsyncComponent(
				() => import('../../../components/player/src/htmlPlayer.vue')
			)
		),
	});

	const props = defineProps({
		viewOption: {
			type: Array<resourceOption>,
			required: true,
		},
	}); 
	const imageOptions = computed(() => { 
		return props.viewOption.filter((m) => m.type == 1);
	});
	

	const videoOptions = computed(() => { 
		return props.viewOption.filter((m) => m.type == 0);
	});

	
	const currentName = ref<'Image' | 'htmlPlayer'>('Image');
	let current_Index =0;
	const current_options = ref<resourceOption>({
		url: '',
		name: 'string',
		type: 1,
		poster: '',
	});
	watch(()=>props.viewOption,(n,o)=>{
		//watch 先于 计算属性
		nextTick(()=>{ 
			if (currentName.value=="Image")current_options.value = imageOptions.value[current_Index];
			else current_options.value = videoOptions.value[current_Index]; 
		}) 
	})
	const onShow = (option: resourceOption, index: number) => { 
		if (option.type == 0) currentName.value = 'htmlPlayer';
		else currentName.value = 'Image';
		current_Index=index
		current_options.value = option;
	}; 
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
