import request from '@/utils/request'

import type { IResponse, ISelect } from 'co6co'
import { create_svc } from 'co6co-right'
import { getBaseUrl } from 'co6co'

const base_URL = '/api/dev'

const { exist_svc, get_select_svc, add_svc, edit_svc, del_svc, get_table_svc } =
  create_svc(base_URL)
export { exist_svc, get_select_svc, add_svc, edit_svc, del_svc, get_table_svc }

export const upload_template = (data: FormData): Promise<IResponse<string>> => {
  return request.post(`${base_URL}/import`, data)
}

export const download_template = (): Promise<IResponse<string>> => {
  return request.get(`${base_URL}/import`)
}

export const getResourceUrl = () => {
  return `${getBaseUrl()}${base_URL}/import`
}
