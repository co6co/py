import {
	ref,
	defineAsyncComponent,
	defineComponent,
	PropType,
	watch,
	onMounted,
} from 'vue';
import { type resourceOption } from './type';
import { ElImage, ElRow, ElCol, ElScrollbar, ElIcon } from 'element-plus';

import { Picture, Loading } from '@element-plus/icons-vue';
import style from '@/assets/css/imageVideo.module.less';
import { loadAsyncResource } from '@/api/download';
import imgStyle from '@/assets/css/images.module.less';
const props = {
	viewOption: {
		type: Object as PropType<Array<resourceOption>>,
		required: true,
	},
} as const;
export default defineComponent({
	name: 'ImageView',
	props: props,
	setup(prop, _) {
		const Image = defineAsyncComponent(() => import('./imageView'));
		const HtmlPlayer = defineAsyncComponent(() => import('./htmlPlayer'));

		/**
     	* 
			const components = reactive({
			Image: markRaw(defineAsyncComponent(() => import('../../../components/player/src/Image.tsx'))),
			htmlPlayer: markRaw(
				defineAsyncComponent(() => import('../../../components/player/src/htmlPlayer.tsx'))
			)
			})
     	*/
		const currentName = ref<'Image' | 'htmlPlayer'>('Image');
		let currentIndex = ref(0);
		const DATA = ref<resourceOption>({
			url: '',
			name: '',
			type: 1,
			poster: '',
		});

		const onShow = (option: resourceOption, index: number) => {
			if (option.type == 0) currentName.value = 'htmlPlayer';
			else currentName.value = 'Image';
			currentIndex.value = index;
			DATA.value = option;
		};
		/**
		 * 导航图数据
		 */
		const posterURlObject = ref<{
			[key: number]: { url: string; authon?: boolean; loading?: boolean };
		}>({});
		const loadImage = (items: resourceOption[]) => {
			items.forEach((item, index) => {
				if (!posterURlObject.value[index])
					posterURlObject.value[index] = { url: '' };
				posterURlObject.value[index].url = '';
				posterURlObject.value[index].authon = item.posterAuthon;
				if (item.posterAuthon)
					loadAsyncResource(item.poster)
						.then((r) => {
							posterURlObject.value[index].url = r;
						})
						.finally(() => {
							posterURlObject.value[index].loading = false;
						});
				else posterURlObject.value[index].url = item.poster;
				if (index == currentIndex.value) onShow(item, index);
			});
		};

		watch(
			() => prop.viewOption,
			(n) => {
				loadImage(n);
			}
		);
		onMounted(() => {
			loadImage(prop.viewOption);
		});
		const onLoadError = (index: number) => {
			if (!posterURlObject.value[index].authon)
				posterURlObject.value[index].loading = false;
		};
		return () => {
			//可以写某些代码
			return (
				<>
					<ElRow class={style.Image}>
						<ElCol span={24}>
							{currentName.value == 'Image' ? (
								<Image option={DATA.value} />
							) : (
								<HtmlPlayer option={DATA.value} />
							)}
						</ElCol>
					</ElRow>
					<ElRow class={style.NavImage}>
						<ElCol span={24}>
							<div class={style.imag_nav_container}>
								<ElScrollbar>
									<ul>
										{prop.viewOption.map((item, index) => {
											return (
												<li onClick={() => onShow(item, index)} key={index}>
													<a href="#">
														<ElImage
															class={imgStyle.image}
															src={posterURlObject.value[index].url}
															title={item.name}
															onError={() => {
																onLoadError(index);
															}}
															style={
																item.type == 0 ? { position: 'relative' } : {}
															}>
															{{
																error: () => {
																	return (
																		<div class="image_slot">
																			{posterURlObject.value[index].authon &&
																			posterURlObject.value[index].loading ? (
																				<ElIcon
																					style="font-size:150%"
																					class="is-loading">
																					<Loading />
																				</ElIcon>
																			) : (
																				<ElIcon style="font-size:150%">
																					<Picture />
																				</ElIcon>
																			)}
																		</div>
																	);
																},
																placeholder: () => {
																	return (
																		<div class="image_slot">
																			<ElIcon
																				style="font-size:150%"
																				class="is-loading">
																				<Loading />
																			</ElIcon>
																		</div>
																	);
																},
															}}
														</ElImage>
													</a>
												</li>
											);
										})}
									</ul>
								</ElScrollbar>
							</div>
						</ElCol>
					</ElRow>
				</>
			);
		};
	},
});
