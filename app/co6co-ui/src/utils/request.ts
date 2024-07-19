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
const parseJson = (response: AxiosResponse) => {
	try {
		let result: IResponse;
		if (typeof response.data == 'string') result = JSONbig.parse(response.data);
		else result = response.data;
		if (result.code == 0) {
			return response.data;
		} else {
			ElMessage.error(`${result.message}`);
			return Promise.reject(response.data);
		}
	} catch (e) {
		return Promise.reject(e);
	}
};
const crateService = (config?: CreateAxiosDefaults<any> | undefined) => {
	const service: AxiosInstance = axios.create(
		config /* {
    baseURL: import.meta.env.VITE_BASE_URL,
    timeout: 5000,
  }*/
	);
	let elLoading: ReturnType<typeof ElLoading.service>;
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
	//增加请求拦截器
	service.interceptors.request.use(
		(config: InternalAxiosRequestConfig) => {
			//发送请求之前
			const token = getToken();
			config.headers.Authorization = `Bearer ${token}`;
			return config;
		},
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		(_error: AxiosError) => {
			//请求错误
			if (elLoading) elLoading.close();
			return Promise.reject();
		}
	);
	//增加响应拦截器
	service.interceptors.response.use(
		(response: AxiosResponse) => {
			//2xx 范围的都会触发该方法
			//if (elLoading) elLoading.close();
			if (response.status === 200) {
				//JSON
				if (response.headers['content-type'] == 'application/json') {
					return parseJson(response);
				} else return response;
			}
			if (response.status === 206) return response;
			else {
				return Promise.reject(response.statusText);
			}
		},
		(error: AxiosError) => {
			//不是 2xx 的触发
			//if (elLoading) elLoading.close();
			if (error.response?.status === 403) {
				removeToken();
				ElMessage.error(`未认证、无权限或者认证信息已过期:${error.message}`);
				//需要验证是否是微信端
				///router.push('/login');
			} else if (error.config && error.config.responseType == 'json') {
				//ElMessage.error(`请求出现错误:${error.message}`);
				ElMessage.error(`请求出现错误:${error.code}`);
			} else if (
				error.config &&
				error.config.headers &&
				error.config.headers['Content-Type'] == 'application/json'
			) {
				//ElMessage.error(`请求出现错误:${error.message}`);
				ElMessage.error(`请求出现错误:${error.code}`);
			}
			return Promise.reject(error);
		}
	);
	return service;
};

const crateService2 = (config?: CreateAxiosDefaults<any> | undefined) => {
	const Axios: AxiosInstance = axios.create(config);
	//增加请求拦截器
	Axios.interceptors.request.use((config: InternalAxiosRequestConfig) => {
		//发送请求之前
		const token = getToken();
		config.headers.Authorization = `Bearer ${token}`;
		return config;
	});
	return Axios;
};

export const createServiceInstance = (timeout: number = 5000) => {
	const baseUrl = getBaseUrl();
	const service: AxiosInstance = crateService({
		baseURL: baseUrl,
		timeout: timeout,
	});
	return service;
};
export const createAxiosInstance = (timeout: number = 5000) => {
	const baseUrl = getBaseUrl();
	const service: AxiosInstance = crateService2({
		baseURL: baseUrl,
		timeout: timeout,
	});
	return service;
};
