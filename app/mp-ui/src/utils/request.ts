import axios, {type AxiosInstance, AxiosError, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios';

import { getToken, removeToken } from './auth';
import { ElLoading, ElMessage } from 'element-plus';
import { useRouter } from 'vue-router';
import router from '../router/index';
import JSONbig from 'json-bigint';

//console.log(import.meta.env)
//console.info(import.meta.env.BASE_URL)
const service: AxiosInstance = axios.create({
	//baseURL: "/v1",//process.env.API_URL_BASE,
	baseURL: import.meta.env.VITE_BASE_URL,
	timeout: 5000,
});
let elLoading: ReturnType<typeof ElLoading.service>;
//增加请求拦截器
service.interceptors.request.use(
	(config: InternalAxiosRequestConfig) => {
		//发送请求之前
		let token = getToken();
		config.headers.Authorization = 'Bearer ' + token;
	 
		return config;
	},
	(error: AxiosError) => {
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
			if (response.headers['content-type'] == 'application/json') {
				if (typeof response.data == 'string') return JSONbig.parse(response.data);
				return response.data;
			} else return response;
		}
		if (response.status === 206) return response;
		else {
			Promise.reject();
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
			ElMessage.error(`请求出现错误:${error.message}`);
		} else if (
			error.config &&
			error.config.headers['Content-Type'] == 'application/json'
		) {
			ElMessage.error(`请求出现错误:${error.message}`);
		}
		return Promise.reject(error);
	}
);

export default service;

const Axios:AxiosInstance = axios.create({ 
    baseURL: import.meta.env.VITE_BASE_URL,
    timeout: 5000, 
});  
//增加请求拦截器
Axios.interceptors.request.use(
	(config: InternalAxiosRequestConfig) => {
		//发送请求之前
		const token = getToken(); 
		config.headers.Authorization = 'Bearer ' + token; 
		return config;
	} 
);  
export {Axios}
