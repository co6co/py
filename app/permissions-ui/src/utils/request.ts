import axios, {type AxiosInstance, AxiosError, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios';
import {getToken,removeToken} from "./auth"
import { ElLoading,ElMessage } from 'element-plus' 
 
import JSONbig  from 'json-bigint'  

const service:AxiosInstance = axios.create({ 
    baseURL: import.meta.env.VITE_BASE_URL,
    timeout: 5000, 
}); 
let elLoading:ReturnType<typeof ElLoading.service >;
 
service.defaults.transformResponse=[(data:any)=>{ 
	return JSONbig.parse(data)
}] 
service.defaults.transformRequest=[(data:any,headers)=>{  
	headers['Content-Type']='application/json;charset=utf-8'  
	return JSONbig.stringify(data) 
}] 
//增加请求拦截器
service.interceptors.request.use(
	(config: InternalAxiosRequestConfig) => {
		//发送请求之前
		const token = getToken(); 
		config.headers.Authorization = 'Bearer ' + token; 
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
			if (response.headers['content-type'] == 'application/json') {
				if (typeof response.data == 'string') return JSONbig.parse(response.data);
				/**
				if (response.data.code==0){
					return response.data
				}else {  
					ElMessage.error(`请求出错：${response.data.message}`)
					return	Promise.reject(response.data.message||"请求出错！"); 
				} 
				 */
				return response.data
			} else return response;
		}
		if (response.status === 206) return response;
		else {

			return 	Promise.reject(response.statusText);
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
		} else if ( error.config && error.config.headers&& error.config.headers['Content-Type'] == 'application/json') {
			//ElMessage.error(`请求出现错误:${error.message}`);
			ElMessage.error(`请求出现错误:${error.code}`);
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