import { createServiceInstance,createAxiosInstance, useRequestToken,type IResponse ,HttpContentType} from 'co6co';
const base_URL = '/api/verify';

export interface IDragVerifyData {
	start: number;
	data: Array<{ t: number; x: number; y: number }>;
	end: number;
}
export const dragVerify_Svc = (
	data: IDragVerifyData
): Promise<IResponse<string>> => {
	return createServiceInstance(3000, false).post(`${base_URL}`, data);
}; 

export const get_captcha_img = (): Promise<any> => {
  return createAxiosNeedToken( HttpContentType.stream).get(`${base_URL}/captcha`, { responseType: 'blob' })
}
export const verify_captcha = (data: { code: string }): Promise<IResponse> => {
  return createServiceInstance(5000,false).post(`${base_URL}/captcha`, data)
} 
export const createAxiosNeedToken = ( 
  responseType: HttpContentType = HttpContentType.stream
) => {
  
  const service = createAxiosInstance(undefined, 30 * 1000, HttpContentType.json, responseType) 
  service.interceptors.request.use(useRequestToken)  
  return service
}