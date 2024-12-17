import { getToken, SessionKey, Storage } from 'co6co'; //, useRequestToken, createAxiosInstance
import { IDownloadConfig } from '@/constants';
import axios, {
	type ResponseType,
	type Method,
	type AxiosRequestConfig,
} from 'axios';

/**
 * 创建 Blob URL 资源
 * @param resource 资源
 * @returns
 */
export const create_URL_resource = (resource: { data: Blob }): string => {
	return URL.createObjectURL(resource.data);
};

/**
 * 加载资源
 * @param url 资源地址
 * @param authon 是否需要认证
 * @param axios_config 配置
 * @returns
 */
const _loadResource = (
	url,
	authon?: boolean,
	axios_config?: AxiosRequestConfig
) => {
	const default_config: {
		method: Method;
		url: string;
		timeout?: number;
		responseType: 'blob';
	} = {
		method: 'get', //请求方式
		responseType: 'blob',
		url: url, //请求地址  会加上 baseURL
		//timeout: 3 * 60 * 1000,
	};
	/**
	const service = createAxiosInstance();
	service.interceptors.request.use(useRequestToken);
	const res = await service.get(url, { ...default_config, ...axios_config });
	 */
	//console.info('Bearer', getToken());
	const store = new Storage();
	const header = authon
		? {
				headers: {
					Authorization: `Bearer ${getToken()}`,
					Session: store.get(SessionKey),
				},
		  }
		: { headers: {} };
	const headers = { ...header.headers, ...axios_config?.headers };
	return axios({
		...default_config,
		...axios_config,
		headers: { ...headers },
	});
};
/**
 * 异步资源载入
 * @param url 资源
 * @param axios_config 配置
 * @returns
 */
export const loadAsyncResource = async (
	url: string,
	axios_config?: AxiosRequestConfig
) => {
	const res = await _loadResource(url, true, axios_config);
	return create_URL_resource({ data: res.data });
};
/**
 *
 * @param url 资源URL
 * @param success 成功回调
 * @param fail 失败回调
 * @param axios_config 配置
 */
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

/**
 *  获取文件长度
 * @param url 资源URL
 * @param authon 认证
 * @returns
 */
export const download_header_svc = (
	url: string,
	authon?: boolean,
	timeout?: number
) => {
	const default_config: {
		method: Method;
		baseURL: string;
		url: string;
		timeout?: number;
	} = {
		baseURL: '',
		url: url, //请求地址  会加上 baseURL
		method: 'HEAD', //请求方式
		timeout: timeout,
	};
	return _loadResource(url, authon, { ...default_config });
};
export const getFileName = (contentDisposition: string) => {
	if (contentDisposition) {
		// Split the Content-Disposition header into parts.
		const parts = contentDisposition.split(';').map((part) => part.trim());
		// Look for the 'filename=' or 'filename*=' part.
		for (let part of parts) {
			if (part.toLowerCase().startsWith('filename=')) {
				// Decode the filename using URL decode, and remove quotes.
				return decodeURIComponent(part.split('=')[1].replace(/"/g, ''));
			} else if (part.toLowerCase().startsWith("filename*=utf-8''")) {
				// Decode the filename using URL decode, and remove the 'UTF-8'' prefix.
				return decodeURIComponent(part.substring("filename*=UTF-8''".length));
			}
		}
	}
	return '未命名';
};
/**
 * 下载文件
 * @param url 资源URL
 * @param config 下载配置  默认时获取文件长度 //{ Range: `bytes=${start}-${end}` },
 * @param authon 是否需要认证
 * @returns
 */
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
		timeout?: number;
	} = {
		method: 'get', //请求方式
		url: url, //请求地址  会加上 baseURL
		responseType: 'blob', //文件流将会被转成blob
		baseURL: '',
		//timeout: 30 * 1000,
	};
	//Object.assign({},default_config,config)
	return _loadResource(url, authon, { ...default_config, ...config });
	//return createAxiosInstance()({ ...default_config, ...config });
};
/**
 * 下载 Blob资源
 * @param resource Blob资源
 */
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

/**
 * 下载资源
 * @param url	资源URL
 * @param fileName 下载文件名
 * @param authon true  增加 token
 * @param finishBck 下载完成回调
 * @param timeout 超时时间
 */
export const download_svc = (
	url: string,
	fileName: string,
	authon?: boolean,
	finishBck?: () => void,
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
			if (finishBck) finishBck();
		});
};
