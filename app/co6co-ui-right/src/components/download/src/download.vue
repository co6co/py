<!--为创建 vue组件示例 不删除-->
<template>
	<el-button
		text
		:icon="Download"
		:loading="obj.downloading"
		@click="onDownload">
		{{
			obj.downloading ? '下载中...' + obj.percentage + '%' : '下载文件'
		}}</el-button
	>
</template>
<script lang="ts" setup>
	import { reactive } from 'vue';
	import { Download } from '@element-plus/icons-vue';
	import {
		download_fragment_svc,
		download_header_svc,
		download_blob_resource,
	} from '@/api/download';
	import { IDownloadConfig } from '@/constants';

	const props = defineProps({
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
	});
	const emit = defineEmits(['downloadpercent']);
	const obj = reactive<{
		downloading: boolean;
		totalSize: number;
		fileBlob: Array<any>;
		percentage: Number;
	}>({
		downloading: false,
		totalSize: 0,
		fileBlob: [],
		percentage: 0,
	});

	const onDownload = async () => {
		obj.fileBlob = [];
		obj.downloading = true;
		obj.totalSize = 0; //文件总大小
		obj.percentage = 0; //下载进度

		download_header_svc(props.url)
			.then(async (res) => {
				const header = res.headers;
				//console.info('contentType:', header)
				obj.totalSize = Number(header['content-length']);
				const contentType = header['content-type'];
				//console.info('contentType:', contentType)
				if (typeof contentType == 'string')
					await startDownload(contentType, props.fileName, props.chunkSize);
			})
			.catch((error) => {
				console.warn('失败', error);
			});
	};
	const download_fragment = async (start: number, end: number) => {
		const config: IDownloadConfig = {
			headers: { Range: `bytes=${start}-${end}` },
		};

		const res = await download_fragment_svc(props.url, config);
		obj.fileBlob.push(res.data);
	};
	const megre_data = (type: string, fileName: string) => {
		//合并
		const blob = new Blob(obj.fileBlob, {
			type: type, //obj.fileBlob[0].type,
		});
		obj.downloading = false;
		download_blob_resource({ data: blob, fileName: fileName });
	};
	const startDownload = async (
		blobType: string,
		fileName: string,
		chunkSize: number
	) => {
		let times = Math.ceil(obj.totalSize / chunkSize);
		//分段下载
		for (let index = 0; index < times; index++) {
			let start = index * chunkSize;
			let end = start + chunkSize - 1;
			if (end >= obj.totalSize) end = obj.totalSize - 1;
			await download_fragment(start, end);
			//计算下载进度
			obj.percentage = Math.floor(((index + 1) / times) * 100);
			emit('downloadpercent', obj.percentage);
			//存储每一片文件流
		}
		megre_data(blobType, fileName);
	};
</script>
