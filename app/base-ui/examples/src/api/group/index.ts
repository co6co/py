import request from '../../utils/request'
import { type IPageResponse, type IResponse, type ITreeSelect } from 'co6co'
import { create_association_svc } from '../'
const BASE_URL = '/api/biz/group'

export interface GroupStatus {
  group: Array<optionItem>
  postion: Array<optionItem>
  allowSetNumberGroup: Array<string>
  allowSetPriorityGroup: Array<string>
}

export const queryList_svc = (data: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/list`, data)
}

export const update_svc = (id: number, data: any): Promise<IResponse> => {
  return request.post(`${BASE_URL}/boatPosNumber/${id}`, data)
}

export const get_status_svc = (): Promise<IPageResponse<GroupStatus>> => {
  return request.get(`${BASE_URL}/getStatus`)
}

export const get_one_svc = (id: number): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/one/${id}`)
}
export const get_select_svc = (): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/select`)
}

export const select_tree_svc = (): Promise<IResponse<ITreeSelect[]>> => {
  return request.post(`${BASE_URL}/selectTree`)
}

export const get_tree_svc = (userId: number): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/tree/${userId}`)
}
export const get_tree_list_by_pid_svc = (pid: number, query: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/list/${pid}`, query)
}
export const get_tree_table_svc = (data: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/tree`, data)
}
//重置所有船的优先级
export const reset_boat_priority = (): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/boat/reset/priority`)
}
//设置船的优先级
export const set_boat_priority = (
  id: number,
  data: { priority: number }
): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/boat/set/priority/${id}`, data)
}
const association_service = create_association_svc(`${BASE_URL}/boat`)
export { association_service }
