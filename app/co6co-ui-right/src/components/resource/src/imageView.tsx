import {
	defineComponent,
	PropType,
	reactive,
	watch,
	nextTick,
	onMounted,
} from 'vue';
import { image2Option } from './type';
import { ElImage, ElEmpty, ElIcon } from 'element-plus';
import { Picture, Loading } from '@element-plus/icons-vue';
import { loadAsyncResource } from '@/api/download';
import imgStyle from '@/assets/css/images.module.less';

const props = {
	option: {
		type: Object as PropType<image2Option>,
		required: true,
	},
} as const;
export default defineComponent({
	name: 'ImageView',
	props: props,
	setup(prop, ctx) {
		const DATA = reactive<image2Option & { loading?: boolean }>(prop.option);
		watch(
			() => prop.option,
			async (n) => {
				await loadResource(n);
			}
		);
		const loadResource = (option?: image2Option) => {
			//DATA.name = n.name
			DATA.loading = true;
			DATA.url = '';
			if (option && ((DATA.authon = option.authon), DATA.authon)) {
				if (option.url)
					nextTick(async () => {
						loadAsyncResource(option.url)
							.then((r) => {
								DATA.url = r;
							})
							.finally(() => {
								DATA.loading = false;
							});
					});
			} else if (option) {
				DATA.url = option.url;
			}
			console.info(DATA, option);
		};
		const onLoadError = () => {
			if (!DATA.authon) DATA.loading = false;
		};
		onMounted(() => {
			console.info('OnMounted', prop.option);
			loadResource(prop.option);
		});

		return () => {
			//可以写某些代码
			return (
				<>
					{prop.option && prop.option.url ? (
						<ElImage
							class={imgStyle.image}
							src={DATA.url}
							previewSrcList={[DATA.url]}
							zoomRate={1.2}
							maxScale={7}
							minScale={0.2}
							onError={onLoadError}
							style={{ height: '100%', width: '100%' }}
							fit="cover">
							{{
								error: () => {
									return (
										<div class="image_slot">
											{DATA.authon && DATA.loading ? (
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
				</>
			);
		};
	},
});
