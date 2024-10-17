import request from '../../utils/request'

import type { IResponse } from 'co6co'
import { create_svc } from 'co6co-right'

const base_URL = '/api/biz/sys/task'
const base_s_URL = '/api/biz/sys/task/sched'

const { exist_svc, add_svc, edit_svc, del_svc, get_table_svc } = create_svc(base_URL)
export { exist_svc, add_svc, edit_svc, del_svc, get_table_svc }

export const exe_once_svc = (id: number): Promise<IResponse> => {
  return request.put(`${base_s_URL}/${id}`)
}

export const exe_sched_svc = (id: number): Promise<IResponse> => {
  return request.post(`${base_s_URL}/${id}`)
}
