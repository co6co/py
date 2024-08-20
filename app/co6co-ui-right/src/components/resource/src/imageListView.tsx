import {
	ref,
	reactive,
	defineComponent,
	watch,
	type PropType,
	onMounted,
	nextTick,
	computed,
} from 'vue';

import { ElImage, ElEmpty, ElIcon } from 'element-plus';
import { Picture, Loading } from '@element-plus/icons-vue';
import imgStyle from '@/assets/css/images.module.less';
import { image2Option } from './type';
import { loadAsyncResource } from '@/api/download';

export default defineComponent({
	name: 'ImagesList',
	props: {
		list: {
			type: Object as PropType<Array<image2Option>>,
			required: true,
		},
	},
	setup(prop) {
		const currentItem = reactive<{
			src?: string;
			url?: string;
			index: number;
			authon?: boolean;
			loading?: boolean;
		}>({
			index: 0,
		});

		const DATA = ref<Array<string>>([]);
		const imageList = computed(() => {
			return DATA.value.map((m) => m);
		});

		watch(
			() => prop.list,
			(n) => {
				currentItem.loading = true;
				if (n && n.length > 0) {
					nextTick(async () => {
						const promises = await prop.list.map(async (item) => {
							return item.authon ? await loadAsyncResource(item.url) : item.url;
						});
						DATA.value = await Promise.all(promises);
						currentItem.loading = false;
						onSwitch(currentItem.index);
					});
				} else {
					DATA.value = [];
				}
				onSwitch(currentItem.index);
			}
		);
		const onSwitch = (value: number) => {
			if (currentItem.index != value) currentItem.index = value;
			if (prop.list && prop.list.length > 0 && prop.list.length > value) {
				currentItem.src = prop.list[value].url;
				currentItem.authon = prop.list[value].authon;
			} else {
				currentItem.src = undefined;
				currentItem.authon = undefined;
			}
			if (DATA.value.length > value) {
				currentItem.url = DATA.value[value];
			}
			if (DATA.value.length == 0) {
				currentItem.url = undefined;
			}
		};
		onMounted(() => {
			onSwitch(currentItem.index);
		});
		const onLoadError = () => {
			if (!currentItem.authon) currentItem.loading = false;
		};
		return () => {
			//可以写某些代码
			return (
				<>
					<div class={imgStyle.imageList}>
						{currentItem.src ? (
							<ElImage
								class={imgStyle.imagesList}
								initialIndex={currentItem.index}
								src={currentItem.url}
								style="width: 100%; height: 100%"
								zoomRate={1.2}
								maxScale={7}
								minScale={0.2}
								fit="cover"
								onError={onLoadError}
								preview-src-list={imageList.value}
								onSwitch={onSwitch}>
								{{
									error: () => {
										return (
											<div class="image_slot">
												{currentItem.authon && currentItem.loading ? (
													<ElIcon style="font-size:200%" class="is-loading">
														<Loading />
													</ElIcon>
												) : (
													<ElIcon>
														<Picture />
													</ElIcon>
												)}
											</div>
										);
									},
									placeholder: () => {
										return (
											<div class="image_slot">
												<ElIcon style="font-size:200%" class="is-loading">
													<Loading />
												</ElIcon>
											</div>
										);
									},
								}}
							</ElImage>
						) : (
							<ElEmpty description="未加载数据" />
						)}
					</div>
				</>
			);
		};
	},
});
