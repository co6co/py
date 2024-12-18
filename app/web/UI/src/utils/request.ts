import { type AxiosInstance } from 'axios'
import { createServiceInstance, requestContentType } from 'co6co'
import { setBaseUrl } from './'
setBaseUrl()
const service: AxiosInstance = createServiceInstance()
export default service

const serviceMultipart = createServiceInstance(15 * 60 * 1000, true, requestContentType.multipart)
export { serviceMultipart }
