import request from '../../utils/request'
const BASE_URL = '/api/biz/dev/aiBox'

export interface Item {
  id: number
  siteId: number
  uuid: string
  code: string
  innerIp: string
  ip: string
  name: string
  cpuNo: string
  mac: string
  license: string
  talkbackNo: string
  createTime: string
  updateTime: string
}

export const select_svc = (): Promise<IResponse> => {
  return request.get(`${BASE_URL}`)
}
export const get = (siteId:number): Promise<IResponse<Item>> => {
  return request.patch(`${BASE_URL}`, {siteId:siteId})
}
export const list = (param: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}`, param)
}
export const add_svc = (data: any): Promise<IResponse> => {
  return request.put(`${BASE_URL}`, data)
}
export const edit_svc = (id: number, data: any): Promise<IResponse> => {
  return request.put(`${BASE_URL}/${id}`, data)
}
export const del_svc = (id: number): Promise<IResponse> => {
  return request.delete(`${BASE_URL}/${id}`)
}
