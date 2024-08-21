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

import { Picture, Loading, CaretRight } from '@element-plus/icons-vue';
import style from '@/assets/css/imageVideo.module.less';
import { loadAsyncResource } from '@/api/download';
import imgStyle from '@/assets/css/images.module.less';
const Image = defineAsyncComponent(() => import('./imageView'));
const HtmlPlayer = defineAsyncComponent(() => import('./htmlPlayer'));

const props = {
	viewOption: {
		type: Object as PropType<Array<resourceOption>>,
		required: true,
	},
} as const;
export default defineComponent({
	name: 'ImageListView',
	props: props,
	setup(prop, _) {
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
			[key: number]: {
				url: string;
				authon?: boolean;
				loading?: boolean;
				succ?: boolean;
			};
		}>({});
		const loadImage = (items: resourceOption[]) => {
			items.forEach((item, index) => {
				if (!posterURlObject.value[index])
					posterURlObject.value[index] = { url: '' };
				posterURlObject.value[index].url = '';
				posterURlObject.value[index].authon = item.posterAuthon;
				posterURlObject.value[index].succ = false;
				if (item.posterAuthon) {
					posterURlObject.value[index].loading = true;
					loadAsyncResource(item.poster)
						.then((r) => {
							posterURlObject.value[index].url = r;
							posterURlObject.value[index].succ = true;
						})
						.finally(() => {
							posterURlObject.value[index].loading = false;
						});
				} else {
					posterURlObject.value[index].url = item.poster;
					posterURlObject.value[index].succ = true;
				}
				//载入当前选中的 内容
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
			//无需认证
			if (!posterURlObject.value[index].authon) {
				posterURlObject.value[index].loading = false;
				posterURlObject.value[index].succ = false;
			}
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
												<li
													onClick={() => onShow(item, index)}
													key={index}
													style={
														item.type == 0 ? { position: 'relative' } : {}
													}>
													<a href="#">
														<ElImage
															class={imgStyle.image}
															src={posterURlObject.value[index].url}
															title={item.name}
															onError={() => {
																onLoadError(index);
															}}>
															{{
																error: () => {
																	return (
																		<div class="image_slot">
																			{posterURlObject.value[index].authon &&
																			posterURlObject.value[index].loading ? (
																				<ElIcon class="is-loading">
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
														{item.type == 0 &&
														!posterURlObject.value[index].loading &&
														posterURlObject.value[index].succ ? (
															<CaretRight />
														) : (
															<></>
														)}
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
