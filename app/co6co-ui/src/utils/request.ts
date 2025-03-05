import axios, {
	type AxiosError,
	type AxiosInstance,
	type AxiosResponse,
	type CreateAxiosDefaults,
	type InternalAxiosRequestConfig,
} from 'axios';
import { ElMessage } from 'element-plus'; //type ElLoading,
import JSONbig from 'json-bigint';
import { getToken, removeToken, getSession } from './auth';
import { getStoreInstance } from '@/hooks';
import { IResponse, HttpContentType } from '@/constants';

/**
 * 获取 apiBaseURL
 * @returns string
 */
export const getBaseUrl = () => {
	const store = getStoreInstance();
	const baseUrl = store.getBaseUrl();
	return baseUrl;
};
const useDefaultResponseBigNumber = (service: AxiosInstance) => {
	service.defaults.transformResponse = [
		(data: any) => {
			return JSONbig.parse(data);
		},
	];
};

/**
 *
 */
export const useRequesContentType = (
	service: AxiosInstance,
	type: string = 'application/json;charset=utf-8'
) => {
	service.defaults.transformRequest = [
		(data: any, headers: any) => {
			headers['Content-Type'] = type;
			if (type.includes('json')) return JSONbig.stringify(data);
			else return data;
		},
	];
};
export const useMultipartRequest = (service: AxiosInstance) => {
	useRequesContentType(service, HttpContentType.multipart);
};
export const useRequestToken = (config: InternalAxiosRequestConfig) => {
	//发送请求之前
	const token = getToken();
	config.headers.Authorization = `Bearer ${token}`;
	useSession(config);
	return config;
};
export const useSession = (config: InternalAxiosRequestConfig) => {
	//发送请求之前
	const session = getSession();
	config.headers.Session = `${session}`;
	return config;
};
export const useText2Json = (response: AxiosResponse) => {
	if (typeof response.data == 'string')
		response.data = JSONbig.parse(response.data);
	return response;
};
export const useResponseJson = (response: AxiosResponse) => {
	if (response.status === 200) {
		//JSON
		if (response.headers['content-type'] == 'application/json') {
			return useText2Json(response);
		} else return response;
	}
	if (response.status === 206) return response;
	else {
		return Promise.reject(response.statusText);
	}
};
const customResponseValid = (
	response: AxiosResponse<IResponse>,
	invalidTip: boolean = false
) => {
	try {
		let result: IResponse = response.data;
		if (result.code == 0) {
			return response.data as any;
		} else {
			if (invalidTip) ElMessage.error(result.message || '处理失败');
			return Promise.reject(response.data);
		}
	} catch (e) {
		return Promise.reject(e);
	}
};
/**
 * 错误提示
 * @param response
 * @returns
 */
export const useResponseValid = (response: AxiosResponse<IResponse>) => {
	return customResponseValid(response, true);
};
/**
 * 错误没有提示
 * @param response
 * @returns
 */
const useResponseValidWithTip = (response: AxiosResponse<IResponse>) => {
	return customResponseValid(response, false);
};
export const createAxios = (config?: CreateAxiosDefaults<any> | undefined) => {
	/* {  baseURL: import.meta.env.VITE_BASE_URL,  timeout: 5000, }*/
	const service: AxiosInstance = axios.create(config);
	return service;
};

/**
 * 带有JSON处理响应处理
 * @param timeout 超时
 * @param tip 	  错误提示
 * @returns
 */
export const createServiceInstance = (
	timeout: number = 5000,
	tip: boolean = true,
	contentType: HttpContentType = HttpContentType.json
) => {
	const baseUrl = getBaseUrl();
	const config = {
		baseURL: baseUrl,
		timeout: timeout,
	};

	//let elLoading: ReturnType<typeof ElLoading.service>;
	const service = createAxios(config);
	useRequesContentType(service, contentType);
	useDefaultResponseBigNumber(service);
	//增加请求拦截器
	service.interceptors.request.use(useRequestToken, (error: AxiosError) => {
		//请求错误
		//if (elLoading) elLoading.close();
		return Promise.reject(error);
	});
	//增加响应拦截器
	service.interceptors.response.use(useResponseJson, (error: AxiosError) => {
		if (error.response?.status === 403) {
			removeToken();
			ElMessage.error(`未认证、无权限或者认证信息已过期:${error.message}`);
		} else if (error.config && error.config.responseType == 'json') {
			ElMessage.error(`请求出现错误:${error.code}`);
		} else if (
			error.config &&
			error.config.headers &&
			error.config.headers['Content-Type'] == 'application/json'
		) {
			ElMessage.error(`请求出现错误:${error.code}`);
		}
		return Promise.reject(error);
	});
	if (tip) service.interceptors.response.use(useResponseValid);
	else service.interceptors.response.use(useResponseValidWithTip);
	return service;
};
/**
 * 没有 有 Authorization head
 * 如需要验证 service.interceptors.request.use(useRequestToken)
 * @param baseUrl
 * @param timeout 超时 ms
 * @param requestContentType
 * @param responseContentType
 * @returns service
 */
export const createAxiosInstance = (
	baseUrl?: string,
	timeout: number = 5000,
	requestContentType: HttpContentType = HttpContentType.json,
	responseContentType: HttpContentType = HttpContentType.json
) => {
	if (!baseUrl) baseUrl = getBaseUrl();
	const config = {
		baseURL: baseUrl,
		timeout: timeout,
	};
	const service: AxiosInstance = createAxios(config);
	useRequesContentType(service, requestContentType);
	if (responseContentType == HttpContentType.json)
		useDefaultResponseBigNumber(service);
	return service;
};
