import request from '../../utils/request'
import * as api_type from 'co6co'
import { create_svc, create_association_svc } from '../'
const base_URL = '/api/users'

export default create_svc(base_URL)

const association_service = create_association_svc(base_URL)
export { association_service }

export const exist_svc = (userName: string, id: number): Promise<api_type.IResponse> => {
  return request.get(`${base_URL}/exist/${userName}/${id}`)
}
export const exist__svc = (data: { userName: string; id: number }): Promise<api_type.IResponse> => {
  return request.post(`${base_URL}/exist`, data)
}
export const retsetPwd_svc = (data: any): Promise<api_type.IResponse> => {
  return request.post(`${base_URL}/reset`, data)
}
export const currentUser_svc = (): Promise<api_type.IResponse> => {
  return request.get(`${base_URL}/currentUser`)
}
export const changePwd_svc = (data: any): Promise<api_type.IResponse> => {
  return request.post(`${base_URL}/changePwd`, data)
}
export interface UserType {
  username: string
  password: string
  role: string
  roleId: string
  permissions: string | string[]
}

export interface UserLogin {
  userName: string
  password: string
}
export const login_svc = (data: UserLogin): Promise<api_type.IResponse> => {
  return request.post(`${base_URL}/login`, data)
}
export const get_state_svc = (): Promise<api_type.IResponse<api_type.IEnumSelect[]>> => {
  return request.post(`${base_URL}/status`)
}
