import request from '../../utils/request'
import { type IPageResponse, type IResponse } from 'co6co'
const BASE_URL = '/api/biz/boat'

export const get_exist_svc = (id: number, name: string): Promise<IPageResponse> => {
  return request.get(`${BASE_URL}/exist/${id}/${name}`)
}
//获取状态列表
export const get_list_svc = (data: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}`, data)
}
export const add_svc = (data: any): Promise<IResponse> => {
  return request.put(`${BASE_URL}`, data)
}
export const edits_svc = (data: any): Promise<IResponse> => {
  return request.patch(`${BASE_URL}`, data)
}
export const edit_svc = (id: number, data: any): Promise<IResponse> => {
  return request.post(`${BASE_URL}/${id}`, data)
}
export const del_svc = (id: number): Promise<IResponse> => {
  return request.delete(`${BASE_URL}/${id}`, {})
}
//检查未关联的船及用户
export const check_svc = (): Promise<IResponse> => {
  return request.post(`${BASE_URL}/check`, {})
}
