
import request from '../../utils/request'
import type {SelectItem} from "../types"
const BASE_URL = '/api/biz/dev/ipCamera'



export interface Item {
  id: number
  uuid:string
  siteId: number
 
  name: string
  code:string
  ptzTopic:string //云台控制主题
  innerIp: string
  ip: string
  sip: string
  channel1_sip: string
  channel2_sip: string
  channel3_sip: string
  channel4_sip: string
  channel5_sip: string
  channel6_sip: string
  streams: String
}

export const select_svc = (siteId: number): Promise<IResponse<SelectItem[]>> => {
  return request.get(`${BASE_URL}?siteId=${siteId}`)
}
/**
 *  通过站点获取列表
 */
export const get = (siteId: number): Promise<IResponse<Item>> => {
  return request.patch(`${BASE_URL}`, { siteId: siteId })
}
export const list = (param: any): Promise<IPageResponse<Item>> => {
  return request.post(`${BASE_URL}`, param)
}
export const add_svc = (data: any): Promise<IPageResponse> => {
  return request.put(`${BASE_URL}`, data)
}
/**
 *  通过Id获取 实体
 */
export const get_svc = (id: number): Promise<IResponse<Item>> => {
  return request.patch(`${BASE_URL}/${id}` )
}

export const edit_svc = (id: number, data: any): Promise<IPageResponse> => {
  return request.put(`${BASE_URL}/${id}`, data)
}

export const del_svc = (id: number): Promise<IPageResponse> => {
  return request.delete(`${BASE_URL}/${id}`)
}

