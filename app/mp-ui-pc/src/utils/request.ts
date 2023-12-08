import axios, {AxiosInstance, AxiosError, AxiosResponse, AxiosRequestConfig} from 'axios';
import {getToken,setToken,removeToken} from "./auth"
import { ElLoading,ElMessage } from 'element-plus' 
import { useRouter } from 'vue-router';
import router from '../router/index';
import JSONbig  from 'json-bigint' 

//console.log(import.meta.env)
//console.info(import.meta.env.BASE_URL)
const service:AxiosInstance = axios.create({
    //baseURL: "/v1",//process.env.API_URL_BASE,
    baseURL: import.meta.env.VITE_BASE_URL,
    timeout: 5000, 
});
let elLoading:ReturnType<typeof ElLoading.service >;
//增加请求拦截器
service.interceptors.request.use(
    (config: AxiosRequestConfig ) => { //发送请求之前  
        if(!config.headers)config.headers={}; 
        config.headers.Authorization="Bearer "+  localStorage.getItem("token" );  
        const noLogin= config.params&&config.params.noLogin
        if(!noLogin) elLoading = ElLoading.service({ fullscreen: true }) 
        else delete config.params.noLogin 
        return config;
    },
    (error: AxiosError) => {//请求错误 
        if(elLoading)elLoading.close() 
        return Promise.reject();
    }
);

//增加响应拦截器
service.interceptors.response.use(
    (response: AxiosResponse) => { //2xx 范围的都会触发该方法
        if(elLoading)elLoading.close();  
        if (response.status === 200 ) { 
            if(response.headers["content-type"]=="application/json")return JSONbig.parse(response.data) ;
            else return response;
        } 
        if (response.status === 206 )  return response;
        else {
            Promise.reject();
        }
    },
    (error: AxiosError) => {//不是 2xx 的触发 
        if(elLoading) elLoading.close(); 
        if(error.response?.status===403){ 
            localStorage.removeItem('ms_username'); 
		    router.push('/login'); 
        }else if ( error.config.responseType =="json") {
            ElMessage.error(`请求出现错误:${error.message}`)
        } 
        return Promise.reject(error);
    }
);

export default service;
