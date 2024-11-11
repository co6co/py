const base_URL = '/api/wx/jsapi' 
import { createServiceInstance, type IResponse } from 'co6co' 
export interface IJsapiResult {
    appId:string
    signature:string
    timestamp:string
    nonceStr:string
}

export const getSignature = (url:string): Promise<IResponse<IJsapiResult>> => {
  return createServiceInstance().post(`${base_URL}`,{url:url})
}
