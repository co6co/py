import { type AxiosInstance } from 'axios'
import { createServiceInstance, getStoreInstance } from 'co6co'
/*
const service: AxiosInstance = crateService({
  baseURL: import.meta.env.VITE_BASE_URL,
  timeout: 5000
})*/
console.info('request.init..')
const store = getStoreInstance()
const baseUrl = import.meta.env.VITE_BASE_URL
store.setBaseUrl(baseUrl)
console.info('baseURL', baseUrl)
const service: AxiosInstance = createServiceInstance()
export default service
