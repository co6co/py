import { defineComponent, PropType, reactive, watch } from 'vue';
import { imageOption } from './type';
import { ElImage, ElEmpty } from 'element-plus';
import { loadAsyncResource } from '@/api/download';

const props = {
	option: {
		type: Object as PropType<imageOption>,
		required: true,
	},
} as const;
export default defineComponent({
	name: 'ImageView',
	props: props,
	setup(prop, _) {
		const DATA = reactive<imageOption>(prop.option);
		watch(
			() => prop.option,
			async (n) => {
				DATA.name = n.name;
				if (n.authon) {
					if (n.url) DATA.url = await loadAsyncResource(n.url);
					else DATA.url = '';
					if (n.poster) DATA.poster = await loadAsyncResource(n.poster);
					else DATA.poster = '';
				} else {
					DATA.url = n.url;
					DATA.poster = n.poster;
				}
			}
		);
		return () => {
			//可以写某些代码
			return (
				<>
					{prop.option.url ? (
						<ElImage
							src={DATA.url}
							previewSrcList={[DATA.url]}
							zoomRate={1.2}
							maxScale={7}
							minScale={0.2}
							style="width: 100%; height: 100%"
							fit="cover"
							title={prop.option.name}
						/>
					) : (
						<ElEmpty v-else description="未加载数据" />
					)}
				</>
			);
		};
	},
});
