import { IFileOption } from '@/constants';

type FileOption = Pick<IFileOption, 'file' | 'subPath'>;
export const useFileDrag = (fileHander: (fileOption: FileOption) => void) => {
	const onFindFile = (file: File, subPath?: string) => {
		const fileOption = { file: file, subPath: subPath };
		fileHander(fileOption);
	};
	/*
	 * 读取目录下的所有文件
	 * 超过只能读取100个文件
	 * @param dirReader 目录读取器 entry.createReader()
	 * @param callback 回调函数 (entries: Array< FileEntry|DirectoryEntry>) => void
	 */
	const _readAllEntries = (
		dirReader: DirectoryReader,
		callback: (entries: Array<FileSystemEntry>) => void
	) => {
		var entries: Array<FileSystemEntry> = [];
		const readFiles = () => {
			dirReader.readEntries(
				(results) => {
					//console.log('results.length: ' + results.length);
					if (results.length > 0) {
						entries.push(...results);
						//console.log('继续读取...');
						readFiles();
					} else callback(entries);
				},
				(error) => {
					console.error('Error reading directory entries:', error);
				}
			);
		};
		readFiles();
	};

	const readFileOrDirectory = (entry: FileSystemEntry) => {
		//entry: FileEntry | DirectoryEntry
		if (entry.isFile) {
			(entry as FileEntry).file((file: File) => {
				// 处理文件 比较慢
				//console.info('处理文件', file.name);
				let subPath = (entry.fullPath as string).replace('/' + file.name, '');
				if (subPath) subPath = subPath.substring(1);
				onFindFile(file, subPath);
			});
		} else if (entry.isDirectory) {
			const dirReader = (entry as DirectoryEntry).createReader();
			_readAllEntries(dirReader, (entries) =>
				entries.forEach((entry) => readFileOrDirectory(entry))
			);
		}
	};
	const onDragOver = (event: DragEvent) => {
		// 阻止默认行为，以允许文件被放置到目标区域
		event.preventDefault();
		event.stopPropagation(); //阻止冒泡
	};

	const onDrop = (event: DragEvent) => {
		// 阻止默认行为，以防止浏览器打开文件
		event.preventDefault();
		event.stopPropagation(); //阻止冒泡
		//console.info('onDrop start...')
		const items = event.dataTransfer?.items;
		if (items && items.length > 0) {
			for (let i = 0; i < items.length; i++) {
				//webkitGetAsEntry 是一个非标准的方法
				const item = items[i].webkitGetAsEntry();
				if (item) {
					readFileOrDirectory(item);
				} else {
					// 如果不是文件系统条目（可能是普通文件）
					const file = event.dataTransfer?.files[i];
					if (file) onFindFile(file);
				}
			}
		} else {
			// 如果 dataTransfer.items 为空，尝试从 dataTransfer.files 获取文件
			/*无法区分文件夹还是目录
			const files = event.dataTransfer?.files;
			if (files && files.length > 0) {
				FileData.value.filePreloadCount = files.length;
				FileData.value.files = Array.from(files).map((f) => {
					return { file: f };
				});
			}
			*/
		}
	};
	return { onDragOver, onDrop };
};
