import {
	ref,
	reactive,
	onMounted,
	defineComponent,
	VNodeChild,
	computed,
} from 'vue';
import { useRoute } from 'vue-router';
import {
	ElCol,
	ElFormItem,
	ElOption,
	ElRow,
	ElScrollbar,
	ElSelect,
} from 'element-plus';
import { file_content_svc } from '@/api/file';
import { create_URL_resource } from '@/api/download';
import { getContypeType, HttpContentType } from 'co6co';
import TextView from '@/components/txtView';
import {
	ImageView,
	HtmlPlayer,
	image2Option,
	videoOption,
} from '@/components/resource';

export default defineComponent({
	name: 'previewView',
	setup(prop, ctx) {
		const pathRef = ref('');
		const DATA = reactive<{
			type: HttpContentType;
			rawContent?: Blob;
			content: image2Option | string | videoOption;
			encording?: string;
		}>({
			type: HttpContentType.text,
			content: '',
			encording: 'UTF-8',
		});
		const parserContent = (blob: Blob) => {
			switch (DATA.type) {
				case HttpContentType.image:
					DATA.content = {
						url: create_URL_resource({
							data: new Blob([blob], { type: HttpContentType.image }),
						}),
						authon: false,
					};

					break;
				case HttpContentType.video:
					DATA.content = {
						url: create_URL_resource({ data: blob }),
						authon: false,
						name: '',
						posterAuthon: false,
						poster: 'void(0)',
					};
					break;
				default:
					DATA.content = '';
					blobToText(blob);
					break;
			}
		};

		const onTypeChanged = (_) => {
			if (DATA.rawContent) parserContent(DATA.rawContent);
		};
		const onEncordingChanged = (_) => {
			if (DATA.rawContent) blobToText(DATA.rawContent);
		};
		const loadData = () => {
			if (pathRef.value)
				file_content_svc(pathRef.value).then((res) => {
					DATA.type = getContypeType(res.headers['content-type']!.toString());
					DATA.rawContent = res.data;
					parserContent(DATA.rawContent!);
				});
		};
		const isText = computed(
			() =>
				DATA.type == HttpContentType.text ||
				DATA.type == HttpContentType.xml ||
				DATA.type == HttpContentType.html ||
				DATA.type == HttpContentType.stream
		);
		const route = useRoute();
		function blobToText(blob) {
			/*
			return new Promise((resolve, reject) => {
				const reader = new FileReader();
				reader.onload = () => resolve(reader.result);
				reader.onerror = reject;
				reader.readAsText(blob); // 默认编码是 UTF-8
			});
			*/
			const reader = new FileReader();
			reader.onload = (event) => {
				DATA.content = event.target!.result as string;
			};
			reader.onerror = (e) => {
				console.error(e);
			};
			reader.readAsText(blob, DATA.encording); // 默认编码是 UTF-8
		}
		onMounted(async () => {
			//console.info('route', route);
			//console.info('state', history.state.params);
			//console.info('router.currentRoute', router.currentRoute);
			const data = history.state.params || route.query; //router.currentRoute.value.params
			pathRef.value = data.path;
			loadData();
		});
		//:page reader

		const rander = (): VNodeChild => {
			return (
				<ElScrollbar>
					<ElRow>
						<ElCol span={10}>
							<ElFormItem label="预览方式">
								<ElSelect v-model={DATA.type} onChange={onTypeChanged}>
									<ElOption label="文本" value={HttpContentType.text} />
									<ElOption label="图片" value={HttpContentType.image} />
									<ElOption label="视频" value={HttpContentType.video} />
									<ElOption label="其他文本" value={HttpContentType.stream} />
								</ElSelect>
							</ElFormItem>
						</ElCol>
						{isText.value ? (
							<ElCol offset={4} span={10}>
								<ElFormItem label="文本编码">
									<ElSelect
										v-model={DATA.encording}
										onChange={onEncordingChanged}>
										<ElOption label="ANSI" value="ANSI" />
										<ElOption label="GB2312" value="GB2312" />
										<ElOption label="GBK" value="GBK" />
										<ElOption label="GB18030" value="GB18030" />
										<ElOption label="UTF-8" value="UTF-8" />
										<ElOption label="统一码UTF-8" value="Unicode" />
									</ElSelect>
								</ElFormItem>
							</ElCol>
						) : (
							<></>
						)}
					</ElRow>

					<ElFormItem label="预览内容">
						{DATA.type == HttpContentType.video ? (
							<HtmlPlayer option={DATA.content} />
						) : DATA.type == HttpContentType.image ? (
							<ImageView style="width:70%;" option={DATA.content} />
						) : (
							<TextView title={pathRef.value} content={DATA.content} />
						)}
					</ElFormItem>
				</ElScrollbar>
			);
		};
		return rander;
	}, //end setup
});
