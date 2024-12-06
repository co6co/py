import request from '@/utils/request'

import type { IResponse } from 'co6co'
const base_URL = '/api/files'
export interface list_param {
  name?: string
  root: string
}
export interface list_res {
  isFile: boolean
  name: string
  path: string
  right: string
  date: string
  size: number
}
export const list_svc = (
  data: list_param
): Promise<IResponse<{ res: list_res[]; root: string }>> => {
  return request.post(`${base_URL}`, data)
}
