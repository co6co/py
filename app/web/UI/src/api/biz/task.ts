import request from '../../utils/request'

import type { IResponse } from 'co6co'
import { create_svc } from 'co6co-right'

const base_URL = '/api/biz/sys/task'
const base_s_URL = '/api/biz/sys/task/sched'
const base_cron_URL = '/api/biz/sys/task/cron/test'
const base_code_URL = '/api/biz/sys/task/code/test'

const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } = create_svc(base_URL)
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc }

export const exe_once_svc = (id: number): Promise<IResponse<string>> => {
  return request.put(`${base_s_URL}/${id}`)
}

export const exe_sched_svc = (id: number): Promise<IResponse> => {
  return request.post(`${base_s_URL}/${id}`)
}
export const stop_sched_svc = (id: number): Promise<IResponse> => {
  return request.delete(`${base_s_URL}/${id}`)
}
export const test_cron_svc = (cron: string): Promise<IResponse<boolean>> => {
  return request.get(`${base_cron_URL}?cron=${cron}`)
}
export const test_cron2_svc = (cron: string): Promise<IResponse<boolean>> => {
  return request.post(`${base_cron_URL}`, { cron: cron })
}

export const test_code_svc = (code: string): Promise<IResponse<boolean>> => {
  return request.post(`${base_code_URL}`, { code: code })
}
export const test_exe_code_svc = (code: string): Promise<IResponse<string>> => {
  return request.put(`${base_code_URL}`, { code: code })
}
