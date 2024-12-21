import request from '@/utils/request'

import { type IResponse, IEnumSelect } from 'co6co'
const base_URL = '/api/tools/num'

export interface param {
  list: number[]
  dans?: number[]
}
export interface category_desc {
  select: number
  z: number
  b: number
  dan: number
}
export const clc_svc = (
  category: number,
  data: param
): Promise<IResponse<Array<Array<number>>>> => {
  return request.post(`${base_URL}/${category}`, data)
}
export const get_category_svc = (): Promise<IResponse<IEnumSelect[]>> => {
  return request.get(`${base_URL}/-1`)
}
export const get_category_desc_svc = (category: number): Promise<IResponse<category_desc>> => {
  return request.get(`${base_URL}/${category}`)
}