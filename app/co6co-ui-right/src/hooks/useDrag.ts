import { ref } from 'vue';
import { IFileOption } from '@/constants';
export const useFileDrag = () => {
	const FileData = ref<{ files: IFileOption[]; filePreloadCount: number }>({
		files: [],
		filePreloadCount: 0,
	});
	/*
	 * 读取目录下的所有文件
	 * 超过只能读取100个文件
	 * @param dirReader 目录读取器 entry.createReader()
	 * @param callback 回调函数 (entries: Array< FileEntry|DirectoryEntry>) => void
	 */
	function _readAllEntries(dirReader, callback: (entries: Array<any>) => void) {
		//FileEntry|DirectoryEntry
		var entries: Array<any> = [];
		const readFiles = () => {
			dirReader.readEntries(
				(results) => {
					if (results.length > 0) {
						entries.push(...results);
						readFiles();
					} else callback(entries);
				},
				(error) => {
					console.error('Error reading directory entries:', error);
				}
			);
		};
		readFiles();
	}
	const readFileOrDirectory = (entry) => {
		//entry: FileEntry | DirectoryEntry
		if (entry.isFile) {
			entry.file((file: File) => {
				let subPath = (entry.fullPath as string).replace('/' + file.name, '');
				if (subPath) subPath = subPath.substring(1);
				FileData.value.files.push({ file: file, subPath: subPath });
			});
		} else if (entry.isDirectory) {
			const dirReader = entry.createReader();
			_readAllEntries(dirReader, (entries) => {
				entries.forEach((entry) => readFileOrDirectory(entry));
			});
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
			FileData.value.filePreloadCount = items.length;
			for (let i = 0; i < items.length; i++) {
				//webkitGetAsEntry 是一个非标准的方法
				const item = items[i].webkitGetAsEntry();
				if (item) {
					readFileOrDirectory(item);
				} else {
					// 如果不是文件系统条目（可能是普通文件）
					const file = event.dataTransfer?.files[i];
					if (file) FileData.value.files.push({ file: file });
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
	return { FileData, onDragOver, onDrop };
};
