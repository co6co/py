import type dragUploader from './dragUploader';
export type DragUploaderInstance = InstanceType<typeof dragUploader>;
export interface IFileOption {
	file: File;
	percentage?: number;
	subPath?: String;
	finished?: boolean;
}
