import { defineComponent, ref, reactive, VNode } from 'vue';
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
	ElDropdown,
	ElDropdownMenu,
	ElDropdownItem,
	ElIcon,
} from 'element-plus';
import {
	ArrowDown,
	Files as FilesIcon,
	Folder as FolderIcon,
} from '@element-plus/icons-vue';
import { upload_svc, get_upload_chunks_svc } from '@/api/file';
import style from '@/assets/css/file.module.less';
import { IFileOption } from './index';
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
			files: Array<IFileOption>;
			uploadFolder: String;
			filePreloadCount: number;
			canelFlag: boolean;
			uploading: boolean;
		}>({
			files: [],
			uploadFolder: '/',
			filePreloadCount: 0,
			canelFlag: false,
			uploading: false,
		});
		const readFileOrDirectory = (entry) => {
			//entry: FileEntry | DirectoryEntry
			if (entry.isFile) {
				entry.file((file: File) => {
					let subPath = (entry.fullPath as string).replace('/' + file.name, '');
					if (subPath) subPath = subPath.substring(1);
					DATA.files.push({ file: file, subPath: subPath });
				});
			} else if (entry.isDirectory) {
				const dirReader = entry.createReader();
				dirReader.readEntries((entries) => {
					entries.forEach((entry) => readFileOrDirectory(entry));
				});
			}
		};
		const onDragOver = (event) => {
			// 阻止默认行为，以允许文件被放置到目标区域
			event.preventDefault();
		};
		const onDrop = (event: DragEvent) => {
			// 阻止默认行为，以防止浏览器打开文件
			event.preventDefault();
			//console.info('onDrop start...')

			const items = event.dataTransfer?.items;
			if (items && items.length > 0) {
				DATA.filePreloadCount = items.length;
				for (let i = 0; i < items.length; i++) {
					//webkitGetAsEntry 是一个非标准的方法
					const item = items[i].webkitGetAsEntry();
					if (item) {
						readFileOrDirectory(item);
					} else {
						// 如果不是文件系统条目（可能是普通文件）
						const file = event.dataTransfer?.files[i];
						if (file) DATA.files.push({ file: file });
					}
				}
			} else {
				// 如果 dataTransfer.items 为空，尝试从 dataTransfer.files 获取文件
				const files = event.dataTransfer?.files;
				if (files && files.length > 0) {
					DATA.filePreloadCount = files.length;
					DATA.files = Array.from(files).map((f) => {
						return { file: f };
					});
				}
			}
			//console.info('onDrop finished.', DATA.files.length)
		};
		const upload = async () => {
			for (let i = 0; i < DATA.files.length; i++) {
				const opt = DATA.files[i];
				if (DATA.canelFlag) return false;
				if (opt.finished) continue;
				const file = opt.file;
				const chunks = createFileChunks(file);
				const uploadedChunks = await getUploadedChunks(
					opt.subPath as string,
					file.name,
					chunks.length
				);

				// 过滤掉已经上传的块
				const remainingChunks = chunks
					.map((v, index) => ({ index: index + 1, value: v }))
					.filter((v) => !uploadedChunks.includes(v.index));
				if (remainingChunks.length === 0) {
					//console.log('所有块已上传完毕')
					opt.percentage = 100;
					continue;
				}
				opt.finished = await uploadFileChunks(
					remainingChunks,
					opt.subPath as string,
					file.name,
					chunks.length,
					(p: number) => {
						opt.percentage = p;
					}
				);
			}
			const unfinshed = DATA.files.filter((o) => !o.finished);
			if (unfinshed.length > 0) return false;
			return true;
		};
		const onUpload = async () => {
			try {
				DATA.uploading = true;
				if (!DATA.files || DATA.files.length == 0) {
					ElMessageBox.alert('请选择要上传的文件或文件夹！');
				}
				DATA.canelFlag = false;
				const result = await upload();
				if (result) {
					ElMessageBox.alert('所有数据已上传完毕!');
					ctx.emit('saved');
				}
			} finally {
				DATA.uploading = false;
			}
		};
		/** 分片上传 */
		const createFileChunks = (file: File, chunkSize = 1 * 1024 * 1024) => {
			const chunks: Array<Blob> = [];
			let start = 0;
			while (start < file.size) {
				const end = Math.min(file.size, start + chunkSize);
				const chunk = file.slice(start, end);
				chunks.push(chunk);
				start = end;
			}
			return chunks;
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

		const onPercentage = (row, column, cellValue: number) => {
			return cellValue ? (cellValue * 100).toFixed(2) + '%' : '';
		};
		/**选择上传 */
		const fileInputRef = ref();
		const onFileSelect = (event) => {
			const files: File[] = Array.from(event.target.files);
			const newData = files.map((file) => {
				const o = { file: file, finished: false };
				return o;
			});
			console.info(newData);
			DATA.files.push(...newData);
		};
		//文件夹
		const folderInputRef = ref();
		const onFolderSelect = (event) => {
			console.info(event);
			const files: File[] = Array.from(event.target.files);
			const newData = files.map((file) => {
				const subPath = file.webkitRelativePath.replace(`/${file.name}`, '');
				const o = { file: file, finished: false, subPath: subPath };
				return o;
			});
			DATA.files.push(...newData);
		};
		const onCommand = (command: string) => {
			switch (command) {
				case 'file':
					fileInputRef.value.click();
					break;
				case 'folder':
					folderInputRef.value.click();
					break;
			}
		};
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
							DATA.files.length > 0 ? 'hasFile' : 'box',
							DATA.canelFlag ? 'canelUpload' : '',
						]}
						v-slots={{
							header: () => (
								<>
									<ElRow>
										<ElCol span={6} class="tl">
											<ElDropdown trigger="click" onCommand={onCommand}>
												{{
													default: () => (
														<ElButton type="primary">
															上传文件或目录
															<ElIcon>
																<ArrowDown />
															</ElIcon>
														</ElButton>
													),
													dropdown: () => (
														<ElDropdownMenu>
															<ElDropdownItem icon={FilesIcon} command="file">
																上传文件
																<input
																	type="file"
																	multiple
																	ref={fileInputRef}
																	style="display: none"
																	onChange={onFileSelect}
																/>
															</ElDropdownItem>
															<ElDropdownItem
																icon={FolderIcon}
																command="folder">
																上传文件夹
																<input
																	type="file"
																	multiple
																	ref={folderInputRef}
																	webkitdirectory
																	style="display: none"
																	onChange={onFolderSelect}
																/>
															</ElDropdownItem>
														</ElDropdownMenu>
													),
												}}
											</ElDropdown>
										</ElCol>
										<ElCol push={12} span={6} class="tr">
											<ElButton
												type="danger"
												onClick={clearFile}
												v-slots={{ default: () => '清空列表' }}
											/>
										</ElCol>
									</ElRow>
								</>
							),
						}}>
						<ElScrollbar onDrop={onDrop} onDragover={onDragOver}>
							{DATA.files.length == 0 ? (
								<div>
									<span class="small">上传文件或文件夹到当前文夹</span>
								</div>
							) : (
								<>
									<ElTable data={DATA.files} ref={tableRef} border={true}>
										<ElTableColumn label="序号" width={55} align="center">
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
									</ElTable>
								</>
							)}
						</ElScrollbar>
					</ElCard>
				</>
			),
		};

		const rander = (): VNode => {
			return <Dialog title={DATA.title} ref={diaglogRef} v-slots={fromSlots} />;
		};
		const hasFile = () => {
			// DATA.files.length 磁盘原因等 延迟很大 需要很久才能出结果
			if (DATA.filePreloadCount > 0) return true;
			return false;
		};
		const openDialog = (folder: string) => {
			DATA.uploadFolder = folder;
			DATA.title = `上传文件或目录至"${folder}"`;
			diaglogRef.value?.openDialog();
		};
		const clearFile = () => {
			DATA.files = [];
			DATA.filePreloadCount = 0;
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
