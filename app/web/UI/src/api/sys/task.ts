import request from '@/utils/request'

import type { IResponse, ISelect } from 'co6co'
import { create_svc } from 'co6co-right'

const base_URL = '/api/sys/task'
const base_s_URL = '/api/sys/task/sched'
const base_cron_URL = '/api/sys/code/cron/test'

const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } = create_svc(base_URL)
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc }

export const exe_once_svc = (id: number): Promise<IResponse<string>> => {
  return request.put(`${base_s_URL}/${id}`)
}
export const get_select_svc = (category: number): Promise<IResponse<ISelect[]>> => {
  return request.get(`${base_URL}/select/${category}`)
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
