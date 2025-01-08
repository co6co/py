import {
	ref,
	reactive,
	onMounted,
	defineComponent,
	VNodeChild,
	computed,
} from 'vue';
import { useRoute } from 'vue-router';
import { ElScrollbar } from 'element-plus';
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
			content: any;
			textContent: string;
		}>({
			type: HttpContentType.text,
			content: '',
			textContent: '',
		});
		const loadData = () => {
			if (pathRef.value)
				file_content_svc(pathRef.value).then((res) => {
					DATA.type = getContypeType(res.headers['content-type']!.toString());
					DATA.content = res.data;
				});
		};
		const route = useRoute();
		//const router = useRouter();
		onMounted(async () => {
			//console.info('route', route);
			//console.info('state', history.state.params);
			//console.info('router.currentRoute', router.currentRoute);
			const data = history.state.params || route.query; //router.currentRoute.value.params
			pathRef.value = data.path;
			loadData();
		});
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
				DATA.textContent = event.target!.result as string;
			};
			reader.readAsText(blob); // 默认编码是 UTF-8
		}
		//:page reader
		const content = computed(() => {
			switch (DATA.type) {
				case HttpContentType.html:
				case HttpContentType.text:
					const res = blobToText(DATA.content);
					return res;
				case HttpContentType.image:
					const data: image2Option = {
						url: create_URL_resource({
							data: new Blob([DATA.content], { type: 'image/jpeg' }),
						}),
						authon: false,
					};
					return data;
				case HttpContentType.video:
					const data2: videoOption = {
						url: create_URL_resource({ data: DATA.content }),
						authon: false,
						name: '',
						posterAuthon: false,
						poster: 'void(0)',
					};
					return data2;
			}
		});
		const rander = (): VNodeChild => {
			return (
				<ElScrollbar>
					{DATA.type == HttpContentType.html ||
					DATA.type == HttpContentType.xml ||
					DATA.type == HttpContentType.text ? (
						<TextView title={pathRef.value} content={DATA.textContent} />
					) : DATA.type == HttpContentType.image ? (
						<ImageView style="width:70%;" option={content.value} />
					) : (
						<HtmlPlayer option={content.value} />
					)}
				</ElScrollbar>
			);
		};
		return rander;
	}, //end setup
});
