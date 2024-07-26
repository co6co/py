import { type AxiosInstance } from 'axios'
import { createServiceInstance } from 'co6co'
import { setBaseUrl } from './'
setBaseUrl()
const service: AxiosInstance = createServiceInstance()
export default service
