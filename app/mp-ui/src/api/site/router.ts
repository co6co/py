import request from '../../utils/request'
const BASE_URL = '/api/biz/dev/router'

export interface Item {
  id: number
  siteId: number
  name: string
  uuid: string
  innerIp: string
  ip: string 
  sim: string
  ssd: string
  password: string
  createTime: string
  updateTime: string
}

export const select_svc = (): Promise<IPageResponse> => {
  return request.get(`${BASE_URL}`)
}
export const get = (siteId: number): Promise<IResponse> => {
  return request.patch(`${BASE_URL}`, { siteId: siteId })
}
export const list = (param: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}`, param)
}
export const add_svc = (data: any): Promise<IPageResponse> => {
  return request.put(`${BASE_URL}`, data)
}
export const edit_svc = (id: number, data: any): Promise<IPageResponse> => {
  return request.put(`${BASE_URL}/${id}`, data)
}
export const del_svc = (id: number): Promise<IPageResponse> => {
  return request.delete(`${BASE_URL}/${id}`)
}
