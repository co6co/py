import { defineComponent, ref, reactive, VNode, computed } from 'vue';
import { Dialog, DialogInstance, byte2Unit } from 'co6co';
import { tableScope } from '@/constants';

import {
	ElButton,
	ElMessageBox,
	ElCard,
	ElScrollbar,
	ElRow,
	ElCol,
	ElTable,
	ElTableColumn,
	ElInputNumber,
} from 'element-plus';
import { Delete as DeleteIcon } from '@element-plus/icons-vue';
import {
	upload_svc,
	createFileChunks,
	get_upload_chunks_svc,
} from '@/api/file';
import style from '@/assets/css/file.module.less';
import { IFileOption } from '@/constants';
import FileDropdown from '@/components/fileDropdown';
import pLimit from 'p-limit';
import { scrollTableToRow } from '@/utils';
import { useFileDrag } from '@/hooks/useDrag';
export default defineComponent({
	name: 'FileUpload',
	props: {},
	emits: {
		//@ts-ignore
		saved: () => true,
	},
	setup(prop, ctx) {
		const diaglogRef = ref<DialogInstance>();
		const DATA = reactive<{
			title?: string;
			uploadFolder: String;
			canelFlag: boolean;
			uploading: boolean;
			limitNum: number;
			scrollIndex: number;
			fileData: IFileOption[];
			hasFile: boolean;
		}>({
			uploadFolder: '/',
			canelFlag: false,
			uploading: false,
			limitNum: 3,
			scrollIndex: 0,
			fileData: [],
			hasFile: false,
		});

		const onFileHander = (file: IFileOption) => {
			DATA.fileData.push(file);
			DATA.hasFile = true;
		};
		const { onDragOver, onDrop } = useFileDrag(onFileHander);

		const hasFile = computed(() => {
			return DATA.hasFile;
		});
		const reset = () => {
			DATA.scrollIndex = 0;
			DATA.hasFile = false;
			DATA.fileData = [];
		};
		const uploadOneFile = async (opt: IFileOption, index: number) => {
			if (opt.finished) return true;
			const file = opt.file;
			let chunks = createFileChunks(file);
			//上传大小为0
			if (chunks.length == 0) {
				chunks = [new Blob()];
			}
			const uploadedChunks = await getUploadedChunks(
				opt.subPath as string,
				file.name,
				chunks.length
			);
			const uploaloadBck = (percentage: number) => {
				opt.percentage = percentage;
				//限制位置为最大
				if (DATA.scrollIndex < index)
					(DATA.scrollIndex = index),
						scrollTableToRow(index, tableRef.value!, scrollbarRef.value!);
			};
			// 过滤掉已经上传的块
			const remainingChunks = chunks
				.map((v, index) => ({ index: index + 1, value: v }))
				.filter((v) => !uploadedChunks.includes(v.index));
			if (remainingChunks.length === 0) {
				//console.log('所有块已上传完毕')
				uploaloadBck(1);
				return true;
			}
			opt.finished = await uploadFileChunks(
				remainingChunks,
				opt.subPath as string,
				file.name,
				chunks.length,
				uploaloadBck
			);
			return opt.finished;
		};
		/**
		 * 开始上传
		 * @returns
		 */
		const uploadAllFile = async () => {
			/*
			for (let i = 0; i < DATA.files.length; i++) {
				const opt = DATA.files[i];
				if (DATA.canelFlag) return false;
				await uploadOne(opt);
			}*/
			const limit = pLimit(DATA.limitNum);
			await Promise.all(
				DATA.fileData.map((opt, index) =>
					limit(() => {
						if (DATA.canelFlag) return false;
						return uploadOneFile(opt, index);
					})
				)
			);
			const unfinshed = DATA.fileData.filter((o) => !o.finished);
			if (unfinshed.length > 0) return false;
			return true;
		};
		const onUpload = async () => {
			try {
				DATA.uploading = true;
				if (!DATA.fileData || DATA.fileData.length == 0) {
					ElMessageBox.alert('请选择要上传的文件或文件夹！');
				}
				DATA.canelFlag = false;
				const result = await uploadAllFile();
				if (result) {
					ElMessageBox.alert('所有数据已上传完毕!');
					ctx.emit('saved');
				}
			} finally {
				DATA.uploading = false;
			}
		};

		const getUploadPath = (root: string, subPath: string) => {
			if (!root) root = '/';
			if ((root.endsWith('/') || root.endsWith('\\')) && subPath)
				return root + subPath;
			else if (subPath) return root + '/' + subPath;
			return root;
		};
		const getUploadedChunks = async (
			subPath: string,
			fileName: string,
			totalChunks: number
		) => {
			const response = await get_upload_chunks_svc({
				fileName: fileName,
				totalChunks: totalChunks,
				uploadPath: getUploadPath(DATA.uploadFolder as string, subPath)!,
			});
			return response.data.uploadedChunks || [];
		};

		const uploadFileChunks = async (
			chunks: Array<{ index: number; value: Blob }>,
			subPath: string,
			fileName: string,
			totalChunks: number,
			bck: (p: number) => void
		) => {
			for (let i = 0; i < chunks.length; i++) {
				if (DATA.canelFlag) return false;
				const chunk = chunks[i];
				const formData = new FormData();
				formData.append('file', chunk.value, `${fileName}_part${chunk.index}`);
				formData.append('index', chunk.index.toString());
				formData.append('totalChunks', totalChunks.toString());
				formData.append('fileName', fileName);
				formData.append(
					'uploadPath',
					getUploadPath(DATA.uploadFolder as string, subPath)!
				);
				try {
					await upload_svc(formData);
					bck((totalChunks - chunks.length + i + 1) / totalChunks);
					//console.log(`块 ${i + 1} 上传成功`, response.message)
				} catch (error) {
					console.error(`块 ${i + 1} 上传失败:`, error);
					// 处理错误，例如重试或提示用户
					return false;
				}
			}
			return true;
		};
		const tableRef = ref<InstanceType<typeof ElTable>>();
		const scrollbarRef = ref<InstanceType<typeof ElScrollbar>>();
		const onPercentage = (row, column, cellValue: number) => {
			return cellValue ? (cellValue * 100).toFixed(2) + '%' : '';
		};
		/**选择上传 */
		//选择文件或文件夹
		const onfileDropdown = (data: IFileOption[]) => {
			DATA.fileData.push(...data);
		};

		const onDelete = (index, _: IFileOption) => {
			//删除从index 的一个元素
			DATA.fileData.splice(index, 1);
		};
		const onLimitNumChange = (n) => {
			if (n <= 0) DATA.limitNum = 3;
		};
		const finshedCount = computed(() => {
			return DATA.fileData.filter((m) => m.finished).length;
		});
		/** end 分片上传 */
		const fromSlots = {
			buttons: () => (
				<>
					<ElButton
						disabled={!DATA.uploading}
						onClick={() => {
							DATA.canelFlag = true;
						}}
						v-slots={{ default: () => '取消上传' }}
					/>
					<ElButton
						disabled={DATA.uploading}
						type={DATA.canelFlag ? 'danger' : 'warning'}
						onClick={onUpload}
						v-slots={{
							default: () => (DATA.canelFlag ? '继续上传' : '开始上传'),
						}}
					/>
				</>
			),
			default: () => (
				<>
					<ElCard
						class={[
							style['drop-Box'],
							DATA.fileData.length > 0 ? 'hasFile' : 'box',
							DATA.canelFlag ? 'canelUpload' : '',
						]}
						v-slots={{
							header: () => (
								<>
									<ElRow>
										<ElCol span={8} class="tl">
											<FileDropdown onSelected={onfileDropdown} />
										</ElCol>
										<ElCol push={1} span={7}>
											<ElInputNumber
												disabled={DATA.uploading}
												v-model={DATA.limitNum}
												placeholder="线程数"
												onChange={onLimitNumChange}
											/>
										</ElCol>
										<ElCol push={1} span={7} class="tr">
											<ElButton
												type="danger"
												onClick={clearFile}
												v-slots={{
													default: () =>
														`清空列表[${finshedCount.value}/${DATA.fileData.length}]`,
												}}
											/>
										</ElCol>
									</ElRow>
								</>
							),
						}}>
						<ElScrollbar
							ref={scrollbarRef}
							onDrop={onDrop}
							onDragover={onDragOver}>
							{DATA.fileData.length == 0 ? (
								<div>
									<span class="small">上传文件或文件夹到当前文夹</span>
								</div>
							) : (
								<>
									<ElTable data={DATA.fileData} ref={tableRef} border={true}>
										<ElTableColumn label="序号" width={112} align="center">
											{{
												default: (scope: tableScope) => scope.$index,
											}}
										</ElTableColumn>
										<ElTableColumn
											label="名称"
											align="center"
											showOverflowTooltip={true}>
											{{
												default: (scope: { row: IFileOption }) =>
													scope.row.file.name,
											}}
										</ElTableColumn>
										<ElTableColumn
											label="文件大小"
											align="center"
											width={130}
											showOverflowTooltip={true}>
											{{
												default: (scope: { row: IFileOption }) =>
													byte2Unit(scope.row.file.size, 'b', 2),
											}}
										</ElTableColumn>
										<ElTableColumn
											prop="percentage"
											formatter={onPercentage}
											label="上传进度"
											align="center"
											width={120}
										/>
										<ElTableColumn
											label="操作"
											width={260}
											align="center"
											fixed="right">
											{{
												default: (scope: {
													row: IFileOption;
													$index: number;
												}) => (
													<ElButton
														disabled={DATA.uploading}
														text={true}
														icon={DeleteIcon}
														onClick={() => onDelete(scope.$index, scope.row)}
														v-slots={{ default: () => '删除' }}
													/>
												),
											}}
										</ElTableColumn>
									</ElTable>
								</>
							)}
						</ElScrollbar>
					</ElCard>
				</>
			),
		};

		const rander = (): VNode => {
			return (
				<Dialog
					closeOnClickModal={false}
					draggable
					title={DATA.title}
					ref={diaglogRef}
					v-slots={fromSlots}
				/>
			);
		};

		const openDialog = (folder: string) => {
			DATA.uploadFolder = folder;
			DATA.title = `上传文件至"${folder}"`;
			diaglogRef.value?.openDialog();
		};
		const clearFile = () => {
			reset();
		};
		ctx.expose({
			openDialog,
			onDragOver,
			onDrop,
			hasFile,
			clearFile,
		});
		rander.openDialog = openDialog;
		rander.onDrop = onDrop;
		rander.onDragOver = onDragOver;
		rander.hasFile = hasFile;
		rander.clearFile = clearFile;
		return rander;
	}, //end setup
});
