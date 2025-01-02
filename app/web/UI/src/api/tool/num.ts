import request from '@/utils/request'

import { type IResponse } from 'co6co'
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
  category: number | string,
  data: param
): Promise<IResponse<{ list: Array<Array<number>>; count: number }>> => {
  return request.post(`${base_URL}/${category}`, data)
}
export const get_category_desc_svc = (category: number): Promise<IResponse<category_desc>> => {
  return request.get(`${base_URL}/${category}`)
}
