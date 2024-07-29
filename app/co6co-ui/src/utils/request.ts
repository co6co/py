import axios, {
	type AxiosError,
	type AxiosInstance,
	type AxiosResponse,
	type CreateAxiosDefaults,
	type InternalAxiosRequestConfig,
} from 'axios';
import { type ElLoading, ElMessage } from 'element-plus';
import JSONbig from 'json-bigint';
import { getToken, removeToken } from './auth';
import { getStoreInstance } from '@/hooks';
import { IResponse } from '@/constants';

/**
 * 获取 apiBaseURL
 * @returns string
 */
export const getBaseUrl = () => {
	const store = getStoreInstance();
	const baseUrl = store.getBaseUrl();
	return baseUrl;
};
const commonHandler = (service: AxiosInstance) => {
	service.defaults.transformResponse = [
		(data: any) => {
			return JSONbig.parse(data);
		},
	];
	service.defaults.transformRequest = [
		(data: any, headers: any) => {
			headers['Content-Type'] = 'application/json;charset=utf-8';
			return JSONbig.stringify(data);
		},
	];
};
export const createAxios = (config?: CreateAxiosDefaults<any> | undefined) => {
	const service: AxiosInstance = axios.create(
		config /* {  baseURL: import.meta.env.VITE_BASE_URL,  timeout: 5000, }*/
	);
	commonHandler(service);
	return service;
};

export const useRequestToken = (config: InternalAxiosRequestConfig) => {
	//发送请求之前
	const token = getToken();
	config.headers.Authorization = `Bearer ${token}`;
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
export const useResponseValid = (response: AxiosResponse<IResponse>) => {
	try {
		let result: IResponse = response.data;
		if (result.code == 0) {
			return response.data as any;
		} else {
			ElMessage.error(`${result.message}`);
			return Promise.reject(response.data);
		}
	} catch (e) {
		return Promise.reject(e);
	}
};

/**
 * 带有JSON处理响应处理
 * @param timeout 超时
 * @returns
 */
export const createServiceInstance = (timeout: number = 5000) => {
	const baseUrl = getBaseUrl();
	const config = {
		baseURL: baseUrl,
		timeout: timeout,
	};

	let elLoading: ReturnType<typeof ElLoading.service>;
	const service = createAxios(config);
	//增加请求拦截器
	service.interceptors.request.use(useRequestToken, (error: AxiosError) => {
		//请求错误
		if (elLoading) elLoading.close();
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
	service.interceptors.response.use(useResponseValid);
	return service;
	return service;
};
/**
 * 有 Authorization head
 * @param timeout 超时 ms
 * @returns
 */
export const createAxiosInstance = (
	baseUrl?: string,
	timeout: number = 5000
) => {
	if (!baseUrl) baseUrl = getBaseUrl();
	const config = {
		baseURL: baseUrl,
		timeout: timeout,
	};
	const service: AxiosInstance = createAxios(config);
	return service;
};
