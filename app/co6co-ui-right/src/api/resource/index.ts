const image_BASE_URL = '/api/res/img';
const video_BASE_URL = '/api/res/video';
const poster_BASE_URL = '/api/res/poster'; // /w/h
const thumbnail_BASE_URL = '/api/res/thumbnail';
const defautlWidth = 200;
const defautlHeidth = 113;
const base_URL = '/api/res/upload';

import {
	getBaseUrl,
	useMultipartRequest,
	createServiceInstance,
	type IResponse,
} from 'co6co';
export const get_img_url = (path: string): string => {
	return `${getBaseUrl()}${image_BASE_URL}?path=${path}`;
};

export const get_video_url = (path: string): string => {
	return `${getBaseUrl()}${video_BASE_URL}?path=${path}`;
};
export const get_thumbnail_url = (
	path: string,
	w: number = defautlWidth,
	h: number = defautlHeidth
): string => {
	return `${getBaseUrl()}${thumbnail_BASE_URL}/${w}/${h}?path=${path}`;
};
export const get_poster_url = (
	path: string,
	w: number = defautlWidth,
	h: number = defautlHeidth
): string => {
	return `${getBaseUrl()}${poster_BASE_URL}/${w}/${h}?path=${path}`;
};

export const img_url = (resourceId: number): string => {
	return `${getBaseUrl()}${image_BASE_URL}/${resourceId}`;
};

export const video_url = (resourceId: number): string => {
	return `${getBaseUrl()}${video_BASE_URL}}/${resourceId}`;
};
export const thumbnail_url = (
	resourceId: number,
	w: number = defautlWidth,
	h: number = defautlHeidth
): string => {
	return `${getBaseUrl()}${thumbnail_BASE_URL}/${resourceId}/${w}/${h}`;
};
export const poster_url = (
	resourceId: number,
	w: number = defautlWidth,
	h: number = defautlHeidth
): string => {
	return `${getBaseUrl()}${poster_BASE_URL}/${resourceId}/${w}/${h}`;
};

export interface IUploadResult {
	resourceId: number;
	path: string;
}
const getService = () => {
	const service = createServiceInstance();
	useMultipartRequest(service);
	return service;
};
const createFileFormData = (file: File) => {
	const formData = new FormData();
	formData.append('file', file, file.name!);
	/*
    const img = new File(['file'], item.file?.name || 'default.jpg', {
      type: item.file!.type
    })*/
	return formData;
};
export const upload_svc = (
	data: any,
	category: string = 'file'
): Promise<IResponse<IUploadResult>> => {
	return getService().put(`${base_URL}/${category}`, data);
};
export const upload_file_svc = (
	data: File
): Promise<IResponse<IUploadResult>> => {
	const formData = createFileFormData(data);
	return upload_svc(formData, 'file');
};

export const upload_image_svc = (
	data: File
): Promise<IResponse<IUploadResult>> => {
	const formData = createFileFormData(data);
	return upload_svc(formData, 'img');
};
export const upload_video_svc = (
	data: File
): Promise<IResponse<IUploadResult>> => {
	const formData = createFileFormData(data);
	return upload_svc(formData, 'video');
};
