import { ref, defineAsyncComponent, defineComponent, PropType } from 'vue';
import { type resourceOption } from './type';
import { ElImage, ElRow, ElCol, ElScrollbar } from 'element-plus';
import style from '@/assets/css/imageVideo.module.less';
import { loadAsyncResource } from '@/api/download';
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

		const posterURlObject = ref<{ [key: number]: string }>({});
		const onloadPosterUrl = (item: resourceOption, index: number) => {
			if (item.authon)
				loadAsyncResource(item.poster).then((r) => {
					posterURlObject.value[index] = r;
				});
			else posterURlObject.value[index] = item.poster;
			return index;
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
															index={onloadPosterUrl(item, index)}
															src={posterURlObject.value[index]}
															title={item.name}
															style={
																item.type == 0 ? { position: 'relative' } : {}
															}></ElImage>
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
