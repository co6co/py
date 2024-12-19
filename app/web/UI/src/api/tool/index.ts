import request from '@/utils/request'

import { type IResponse } from 'co6co'
const base_URL = '/api/tools/num'

export interface param {
  list: number[]
  dans?: number[]
}
export const clc_svc = (
  category: number,
  data: param
): Promise<IResponse<Array<Array<number>>>> => {
  return request.post(`${base_URL}/${category}`, data)
}
