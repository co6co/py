import request from '@/utils/request'

import { getBaseUrl, type IResponse } from 'co6co'
import { download_svc } from 'co6co-right'
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
export const getResourceUrl = (filePath: string, isFile: boolean) => {
  //return 'http://127.0.0.1/co6co-0.0.1.tgz'
  if (isFile) return `${getBaseUrl()}${base_URL}?path=${encodeURIComponent(filePath)}`
  else return `${getBaseUrl()}${base_URL}/zip?path=${encodeURIComponent(filePath)}`
}
export const downFile_svc = (filePath: string, fileName: string) => {
  download_svc(`${getBaseUrl()}${base_URL}?path=${encodeURIComponent(filePath)}`, fileName, true)
}
