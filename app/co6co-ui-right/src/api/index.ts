import { createServiceInstance, createAxiosInstance } from 'co6co';
import { IDownloadConfig } from '@/constants';
import { type ResponseType, type Method, type AxiosRequestConfig } from 'axios';
import * as api_type from 'co6co';

/*
export const fetchData = () => {
	return createService()({
		url: './table.json',
		method: 'get',
	});
};
*/
//创建 Blob 资源
export const create_URL_resource = (resource: { data: Blob }): string => {
	return URL.createObjectURL(resource.data);
};
//下载  blob 资源
export const download_blob_resource = (resource: {
	data: Blob;
	fileName: string;
}) => {
	const link = document.createElement('a');
	link.href = create_URL_resource({ data: resource.data });
	link.download = resource.fileName;
	link.click();
	window.URL.revokeObjectURL(link.href);
};
export const request_resource_svc = async (
	url: string,
	axios_config: AxiosRequestConfig = { method: 'get', responseType: 'blob' }
) => {
	const default_config: { method: Method; url: string; timeout: number } = {
		method: 'get', //请求方式
		url: url, //请求地址  会加上 baseURL
		timeout: 3 * 60 * 1000,
	};
	const res = await createServiceInstance()({
		...default_config,
		...axios_config,
	});
	//const blob = new Blob([res.data]);//处理文档流
	const result = create_URL_resource({ data: res.data });
	return result;
};

//下载文件
//download_config 为默认时获取文件长度
export const download_header_svc = (url: string) => {
	const default_config: {
		method: Method;
		baseURL: string;
		url: string;
		timeout: number;
	} = {
		baseURL: '',
		url: url, //请求地址  会加上 baseURL
		method: 'HEAD', //请求方式
		timeout: 30 * 1000,
	};
	return createAxiosInstance()({ ...default_config });
};
//下载文件
//download_config 为默认时获取文件长度
export const download_fragment_svc = (
	url: string,
	config: IDownloadConfig = { method: 'HEAD' }
) => {
	const default_config: {
		method: Method;
		baseURL: string;
		url: string;
		responseType: ResponseType;
		timeout: number;
	} = {
		method: 'get', //请求方式
		url: url, //请求地址  会加上 baseURL
		responseType: 'blob', //文件流将会被转成blob
		baseURL: '',
		timeout: 30 * 1000,
	};
	//Object.assign({},default_config,config)
	return createAxiosInstance()({ ...default_config, ...config });
};
//单独下载
//需要认证的下载
export const download_svc = (
	url: string,
	fileName: string,
	bck?: () => void
) => {
	createAxiosInstance()
		.get(url, {
			method: 'get', //请求方式
			responseType: 'blob', //文件流将会被转成blob
			timeout: 60 * 1000,
		})
		.then((res) => {
			try {
				//console.info(res,res)
				const blob = new Blob([res.data]); //处理文档流
				const down = document.createElement('a');
				down.download = fileName;
				down.style.display = 'none'; //隐藏,没必要展示出来
				down.href = URL.createObjectURL(blob);
				document.body.appendChild(down);
				down.click();
				URL.revokeObjectURL(down.href); // 释放URL 对象
				document.body.removeChild(down); //下载完成移除
			} catch (e) {
				console.warn('catchError', e);
			}
		})
		.catch((e) => {
			console.error('e:', e);
		})
		.finally(() => {
			if (bck) bck();
		});
};

export const create_svc = (baseUrl: string) => {
	const get_select_svc = (
		data?: any
	): Promise<api_type.IResponse<api_type.ISelect[]>> => {
		return createServiceInstance().get(`${baseUrl}`, data);
	};
	const get_table_svc = (data: any): Promise<api_type.IPageResponse> => {
		return createServiceInstance().post(`${baseUrl}`, data);
	};
	const exist_svc = (code: string): Promise<api_type.IResponse> => {
		return createServiceInstance().post(`${baseUrl}/exist/${code}`);
	};
	const add_svc = (data: any): Promise<api_type.IResponse> => {
		return createServiceInstance().put(`${baseUrl}`, data);
	};
	const edit_svc = (id: number, data: any): Promise<api_type.IResponse> => {
		return createServiceInstance().put(`${baseUrl}/${id}`, data);
	};
	const del_svc = (id: number): Promise<api_type.IResponse> => {
		return createServiceInstance().delete(`${baseUrl}/${id}`, {});
	};
	return {
		get_select_svc,
		get_table_svc,
		exist_svc,
		add_svc,
		edit_svc,
		del_svc,
	};
};
export const create_tree_svc = (baseUrl: string) => {
	const get_select_tree_svc = (
		key?: number | string
	): Promise<api_type.IPageResponse> => {
		if (key != undefined)
			return createServiceInstance().get(`${baseUrl}/tree/${key}`);
		else return createServiceInstance().get(`${baseUrl}/tree`);
	};
	const get_tree_table_svc = (data: any): Promise<api_type.IPageResponse> => {
		return createServiceInstance().post(`${baseUrl}/tree`, data);
	};
	const service = create_svc(baseUrl);
	return { get_select_tree_svc, get_tree_table_svc, ...service };
};

export const create_association_svc = (baseUrl: string) => {
	const get_association_svc = (
		associationValue: number | string,
		data: any
	): Promise<api_type.IResponse> => {
		return createServiceInstance().post(
			`${baseUrl}/association/${associationValue}`,
			data
		);
	};
	const save_association_svc = (
		associationValue: number | string,
		data: api_type.IAssociation
	): Promise<api_type.IResponse> => {
		return createServiceInstance().put(
			`${baseUrl}/association/${associationValue}`,
			data
		);
	};
	return { get_association_svc, save_association_svc };
};
