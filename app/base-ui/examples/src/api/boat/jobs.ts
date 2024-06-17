import request from '../../utils/request'
import type { IPageResponse, IResponse, IEnumSelect } from 'co6co'
const BASE_URL = '/api/biz/job'

export const get_exist_svc = (id: number, name: string): Promise<IPageResponse> => {
  return request.get(`${BASE_URL}/exist/${id}/${name}`)
}
/**
 * 获取状态列表
 * */
export const get_state_svc = (): Promise<IResponse<{ stateslist: IEnumSelect[] }>> => {
  return request.get(`${BASE_URL}`)
}
//获取状态列表
export const get_list_svc = (data: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}`, data)
}
export const add_svc = (data: any): Promise<IResponse> => {
  return request.put(`${BASE_URL}`, data)
}

export interface JobItem {
  id: number //": 105006702497498226,
  status: number //": 2,
  recordingIds: string //": "1444,1445,1446,1447,1448,1449,1450,1451,1452,1453",
  createUser: number //": null,
  createTime: string //": "2024-04-09 14:16:22",
  userId: number //": 7,
  recordIds: string //": "1444,1445,1446,1447,1448,1449,1450,1451,1452,1453,1454,1455,1456,1457,1458,1459,1460",
  jobId: number //": 173139387475537912,
  timeOut: number //": 20,
  updateUser: string //": null,
  updateTime: string //": "2024-04-09 14:54:58"
}
/**
 * 查询任务包含的 记录
 * @param id 主键
 * @returns
 */
export const get_records_svc = (
  id: BigInt
): Promise<IResponse<{ list: []; job: JobItem; unAuditIds: number[] }>> => {
  return request.get(`${BASE_URL}/${id}`)
}
export const edit_svc = (id: number, data: any): Promise<IResponse> => {
  return request.post(`${BASE_URL}/${id}`, data)
}
export const del_svc = (id: number): Promise<IResponse> => {
  return request.delete(`${BASE_URL}/${id}`, {})
}
