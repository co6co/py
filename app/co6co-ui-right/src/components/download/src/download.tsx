import {
	defineComponent,
	SlotsType,
	Slot,
	VNode,
	reactive,
	computed,
} from 'vue';

import { Download } from '@element-plus/icons-vue';
import { ElButton } from 'element-plus';
import {
	download_fragment_svc,
	download_blob_resource,
	download_header_svc,
} from '@/api/index';
import { IDownloadConfig } from '@/constants';
//定义属性
const props = {
	url: {
		type: String,
		required: true,
	},
	chunkSize: {
		type: Number,
		default: 5 * 1024 * 1024,
	},
	fileName: {
		type: String,
		required: true,
	},
	//是否展示进度条
	showPercentage: {
		type: Boolean,
		default: true,
	},
} as const;

//定义事件
interface Emits {
	(e: 'downloadpercent', percentage: number): void;
}
export interface DownloadSlots {
	default?: Slot<{ title: string }>;
}
export default defineComponent({
	props: props,
	emits: ['downloadpercent'] as const,
	//slots: ['default'] as const,
	slots: Object as SlotsType<DownloadSlots>,
	setup(prop, { emit, slots }: { emit: Emits; slots: DownloadSlots }) {
		const DATA = reactive<{
			downloading: boolean;
			totalSize: number;
			fileBlob: Array<any>;
			percentage: number;
		}>({
			downloading: false,
			totalSize: 0,
			fileBlob: [],
			percentage: 0,
		});

		const onDownload = async () => {
			DATA.fileBlob = [];
			DATA.downloading = true;
			DATA.totalSize = 0; //文件总大小
			DATA.percentage = 0; //下载进度

			download_header_svc(prop.url)
				.then(async (res) => {
					const header = res.headers;
					//console.info('contentType:', header)
					DATA.totalSize = Number(header['content-length']);
					const contentType = header['content-type'];
					//console.info('contentType:', contentType)
					if (typeof contentType == 'string')
						await startDownload(contentType, prop.fileName, prop.chunkSize);
				})
				.catch((error) => {
					console.warn('失败', error);
				});
		};
		const download_fragment = async (start: number, end: number) => {
			const config: IDownloadConfig = {
				headers: { Range: `bytes=${start}-${end}` },
			};

			const res = await download_fragment_svc(prop.url, config);
			DATA.fileBlob.push(res.data);
		};
		const megre_data = (type: string, fileName: string) => {
			//合并
			const blob = new Blob(DATA.fileBlob, {
				type: type, //DATA.fileBlob[0].type,
			});
			DATA.downloading = false;
			download_blob_resource({ data: blob, fileName: fileName });
		};
		const startDownload = async (
			blobType: string,
			fileName: string,
			chunkSize: number
		) => {
			let times = Math.ceil(DATA.totalSize / chunkSize);
			//分段下载
			for (let index = 0; index < times; index++) {
				let start = index * chunkSize;
				let end = start + chunkSize - 1;
				if (end >= DATA.totalSize) end = DATA.totalSize - 1;
				await download_fragment(start, end);
				//计算下载进度
				DATA.percentage = Math.floor(((index + 1) / times) * 100);
				emit('downloadpercent', DATA.percentage);
				//存储每一片文件流
			}
			megre_data(blobType, fileName);
		};
		const defaultText = computed(() => {
			return DATA.downloading
				? '下载中...' + DATA.percentage + '%'
				: '下载文件';
		});
		const rander = (): VNode => {
			return (
				<ElButton
					text={true}
					icon={Download}
					loading={DATA.downloading}
					onClick={onDownload}>
					{slots.default ? slots.default : defaultText.value}
				</ElButton>
			);
		};
		/*
		const openDialog = (oper: FormOperation, item?: Item) => {
			init_data(oper, item);
			diaglogForm.value?.openDialog();
		};
		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
        */
		return rander;
	}, //end setup
});
