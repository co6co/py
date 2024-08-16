import {
	ref,
	reactive,
	defineComponent,
	watch,
	type PropType,
	onMounted,
	computed,
} from 'vue';

import { ElImage, ElEmpty, ElIcon } from 'element-plus';
import { Picture } from '@element-plus/icons-vue';
import imgStyle from '@/assets/css/images.module.less';
import { image2Option } from './type';
import { loadAsyncResource } from '@/api/download';

export default defineComponent({
	name: 'ImagesList',
	props: {
		list: {
			//请求的url会加上 baseURL
			type: Object as PropType<Array<image2Option>>,
			required: true,
		},
	},
	setup(prop) {
		const currentItem = reactive<image2Option>({
			url: '',
		});
		const DATA = ref<Array<image2Option>>([]);
		const imageList = computed(() => {
			return DATA.value.map((m) => m.url);
		});
		const index = ref(0);
		watch(
			() => prop.list,
			async (n) => {
				if (n && n.length > 0) {
					const promises = await prop.list.map(async (item) => {
						return {
							url: item.authon ? await loadAsyncResource(item.url) : item.url,
							authon: item.authon,
						};
					});
					DATA.value = await Promise.all(promises);
				}
				if (DATA.value.length > 0)
					currentItem.url = DATA.value[index.value].url;
			}
		);
		const onSwitch = (value: number) => {
			index.value = value;
			if (DATA.value.length > index.value)
				currentItem.url = DATA.value[index.value].url;
		};
		onMounted(() => {
			onSwitch(index.value);
		});
		return () => {
			//可以写某些代码
			return (
				<>
					<div class={imgStyle.imageList}>
						{currentItem.url ? (
							<ElImage
								class={imgStyle.imagesList}
								initialIndex={index.value}
								src={currentItem.url}
								style="width: 100%; height: 100%"
								zoomRate={1.2}
								maxScale={7}
								minScale={0.2}
								fit="cover"
								preview-src-list={imageList}
								onSwitch={onSwitch}
								v-slots={{
									error: () => {
										return (
											<div class={imgStyle.image_slot}>
												<ElIcon>
													<Picture />
												</ElIcon>
											</div>
										);
									},
								}}
							/>
						) : (
							<ElEmpty description="未加载数据" />
						)}
					</div>
				</>
			);
		};
	},
});
