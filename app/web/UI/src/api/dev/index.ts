import request from '@/utils/request'

import type { IResponse, IEnumSelect } from 'co6co'
import { create_svc } from 'co6co-right'
import { getBaseUrl, createServiceInstance, HttpContentType } from 'co6co'

const base_URL = '/api/dev'
const base_img_URL = '/api/dev/img'

const { exist_svc, get_select_svc, add_svc, edit_svc, del_svc, get_table_svc } =
  create_svc(base_URL)
export { exist_svc, get_select_svc, add_svc, edit_svc, del_svc, get_table_svc }

export const upload_template = (data: FormData): Promise<IResponse<string>> => {
  return createServiceInstance(30 * 1000, true, HttpContentType.form).post(
    `${base_URL}/import`,
    data
  )
}
export const download_template = (): Promise<IResponse<string>> => {
  return request.get(`${base_URL}/import`)
}
export const dev_category_svc = (): Promise<IResponse<Array<IEnumSelect>>> => {
  return request.get(`${base_URL}/category`)
}

export const getResourceUrl = () => {
  return `${getBaseUrl()}${base_URL}/import`
}
export const getCheckDataUrl = () => {
  return `${getBaseUrl()}${base_URL}/downCheckData`
}
export const img_folder_select_svc = (): Promise<IResponse<Array<string>>> => {
  return request.get(`${base_img_URL}`)
}
export const img_list_svc = (data: { date: string }): Promise<IResponse<Array<string>>> => {
  return request.post(`${base_img_URL}`, data)
}
export const img_preview_svc = (date: string, name: string): Promise<IResponse<Array<string>>> => {
  return request.get(`${base_img_URL}/preview/${date}/${name}`)
}
export const img_del_folder_svc = (date: string): Promise<IResponse> => {
  return request.delete(`${base_img_URL}/${date}`)
}
