import { getToken, useRequestToken, createAxiosInstance } from 'co6co';
import { IDownloadConfig } from '@/constants';
import axios, {
	type ResponseType,
	type Method,
	type AxiosRequestConfig,
} from 'axios';

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
const _loadResource = (
	url,
	authon?: boolean,
	axios_config?: AxiosRequestConfig
) => {
	const default_config: {
		method: Method;
		url: string;
		timeout: number;
		responseType: 'blob';
	} = {
		method: 'get', //请求方式
		responseType: 'blob',
		url: url, //请求地址  会加上 baseURL
		timeout: 3 * 60 * 1000,
	};
	/**
	const service = createAxiosInstance();
	service.interceptors.request.use(useRequestToken);
	const res = await service.get(url, { ...default_config, ...axios_config });
	 */
	//console.info('Bearer', getToken());
	const header = authon
		? { headers: { Authorization: `Bearer ${getToken()}` } }
		: { headers: {} };
	return axios({
		...header,
		...default_config,
		...axios_config,
	});
};
export const loadAsyncResource = async (
	url: string,
	axios_config?: AxiosRequestConfig
) => {
	const res = await _loadResource(url, true, axios_config);
	return create_URL_resource({ data: res.data });
};

export const loadResource = (
	url,
	success: (url: string) => void,
	fail: (e: string) => void,
	axios_config?: AxiosRequestConfig
) => {
	_loadResource(url, true, axios_config)
		.then((res) => {
			if (res.status == 200) {
				//const blob = new Blob([res.data]) //处理文档流
				const result = create_URL_resource({ data: res.data });
				success ? success(result) : console.info(result);
			} else {
				fail ? fail(res.data) : console.warn(`请求：${url}error`, res.data);
			}
		})
		.catch((e) => {
			fail ? fail(e) : console.warn(`请求：${url}error`, e);
		});
};

//下载文件
//download_config 为默认时获取文件长度
export const download_header_svc = (url: string, authon?: boolean) => {
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
	return _loadResource(url, authon, { ...default_config });
};
//下载文件
//download_config 为默认时获取文件长度
export const download_fragment_svc = (
	url: string,
	config: IDownloadConfig = { method: 'HEAD' },
	authon?: boolean
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
	return _loadResource(url, authon, { ...default_config, ...config });
	//return createAxiosInstance()({ ...default_config, ...config });
};
//单独下载
//需要认证的下载
export const download_svc = (
	url: string,
	fileName: string,
	authon?: boolean,
	bck?: () => void,
	timeout?: number
) => {
	_loadResource(url, authon, { timeout: timeout || 60 * 1000 })
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
