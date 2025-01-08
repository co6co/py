import { type AxiosInstance } from 'axios'
import { createServiceInstance, HttpContentType } from 'co6co'
import { setBaseUrl } from './'
setBaseUrl()
const service: AxiosInstance = createServiceInstance()
export default service

const serviceMultipart = createServiceInstance(15 * 60 * 1000, true, HttpContentType.multipart)
export { serviceMultipart }
