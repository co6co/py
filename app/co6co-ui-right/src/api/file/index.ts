import {
	createServiceInstance,
	getBaseUrl,
	type IResponse,
	requestContentType,
} from 'co6co';

import { download_svc } from '@/api/download';
const base_URL = '/api/files';
export interface list_param {
	name?: string;
	root: string;
}
export interface list_res {
	isFile: boolean;
	name: string;
	path: string;
	right: string;
	date: string;
	size: number;
}
const createMultipartService = () => {
	return createServiceInstance(
		15 * 60 * 1000,
		true,
		requestContentType.multipart
	);
};
/**
 * 文件列表
 * @param data
 * @returns
 */
export const list_svc = (
	data: list_param
): Promise<IResponse<{ res: list_res[]; root: string }>> => {
	return createServiceInstance().post(`${base_URL}`, data);
};
/**
 * 获取资源URL
 * @param filePath
 * @param isFile
 * @returns url
 */
export const getResourceUrl = (filePath: string, isFile: boolean) => {
	//return 'http://127.0.0.1/co6co-0.0.1.tgz'
	if (isFile)
		return `${getBaseUrl()}${base_URL}?path=${encodeURIComponent(filePath)}`;
	else
		return `${getBaseUrl()}${base_URL}/zip?path=${encodeURIComponent(
			filePath
		)}`;
};
export const downFile_svc = (filePath: string, fileName: string) => {
	download_svc(
		`${getBaseUrl()}${base_URL}?path=${encodeURIComponent(filePath)}`,
		fileName,
		true
	);
};
/**
 * 删除文件
 * @param path 路径路径
 * @returns
 */
export const del_svc = (path: string): Promise<IResponse> => {
	return createServiceInstance().delete(`${base_URL}?path=${path}`);
};

/**
 * 上传文件- 支持断点续传
 * @param data
 * @returns
 */
export const upload_svc = (data: FormData): Promise<IResponse> => {
	return createMultipartService().put(`${base_URL}/upload`, data);
};
/**
 * 查询已上传的 chunks
 * @param data 参数
 * @returns
 */
export const get_upload_chunks_svc = (data: {
	fileName: string;
	totalChunks: number;
	uploadPath: string;
}): Promise<IResponse<{ uploadedChunks: Array<number> }>> => {
	return createServiceInstance().post(`${base_URL}/upload/query`, data);
};
