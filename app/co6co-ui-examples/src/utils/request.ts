import axios, { type AxiosInstance, type InternalAxiosRequestConfig } from 'axios'
import { createAxiosInstance, getToken } from 'co6co'
/*
const service: AxiosInstance = crateService({
  baseURL: import.meta.env.VITE_BASE_URL,
  timeout: 5000
})*/
const service: AxiosInstance = createAxiosInstance()
export default service
const Axios: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_BASE_URL,
  timeout: 5000
})
//增加请求拦截器
Axios.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  //发送请求之前
  const token = getToken()
  config.headers.Authorization = `Bearer ${token}`
  return config
})
export { Axios }
