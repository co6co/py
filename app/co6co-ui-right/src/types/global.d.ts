declare global {
	// FileEntry 表示文件条目
	interface FileEntry extends FileSystemEntry {
		file(
			successCallback: (file: File) => void,
			errorCallback?: (error: FileError) => void
		): void;
		createWriter(
			successCallback: (writer: FileWriter) => void,
			errorCallback?: (error: FileError) => void
		): void;
	}

	// DirectoryEntry 表示目录条目
	interface DirectoryEntry extends FileSystemEntry {
		isFile: false;
		isDirectory: true;

		// 创建一个 DirectoryReader 来读取目录中的条目
		createReader(): DirectoryReader;

		// 在当前目录下获取或创建子目录
		getDirectory(
			path: string,
			options: Flags,
			successCallback: (directory: DirectoryEntry) => void,
			errorCallback?: (error: FileError) => void
		): void;

		// 在当前目录下获取或创建文件
		getFile(
			path: string,
			options: Flags,
			successCallback: (file: FileEntry) => void,
			errorCallback?: (error: FileError) => void
		): void;
	}

	// DirectoryReader 用于读取目录内容
	interface DirectoryReader {
		readEntries(
			successCallback: (entries: FileSystemEntry[]) => void,
			errorCallback?: (error: FileError) => void
		): void;
	}
	/**
	// Flags 用于指定操作选项（如创建或覆盖）
	interface Flags {
		create?: boolean;
		exclusive?: boolean;
	}
	// Metadata 包含文件或目录的元数据
	interface Metadata {
		size: number; // 文件大小（仅适用于文件）
		modificationTime: Date; // 修改时间
	}

	// FileError 表示文件操作中的错误
	interface FileError {
		code: number;
	}

	//FileSystem 表示文件系统
	interface FileSystem {
		name: string; // 文件系统的名称
		root: DirectoryEntry; // 文件系统的根目录
	}
 */
	// Entry 是 FileEntry 和 DirectoryEntry 的联合类型
	type Entry = FileEntry | DirectoryEntry;
}

export {};
