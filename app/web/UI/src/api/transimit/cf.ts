import request from '@/utils/request'
import { type IResponse, IEnumSelect } from 'co6co'

const base_url = '/api/transmit/cf'
export interface IParam {
  record_id?: string //: 创建时不需要,
  type: string
  name: string //: "www", # 子域名
  content: string //: "1.1.1.1",
  ttl: number //: 1,
  proxied: boolean //: true
  comment?: string
}

export enum RecordType {
  A = 'A',
  AAAA = 'AAAA',
  CNAME = 'CNAME',
  NS = 'NS',
  TXT = 'TXT',
  SRV = 'SRV',
  LOC = 'LOC',
  MX = 'MX'
}
export interface IListItem {
  id: string
  name: string
  type: string
  content: string
  proxiable: boolean
  proxied: boolean
  ttl: number
  settings: {}
  meta: {
    origin_worker_id: string
  }
  comment: null
  tags: []
  created_on: string
  modified_on: string
}
interface IResult_info {
  page: number
  per_page: number
  count: number
  total_count: number
  total_pages: number
}
export interface IListResult {
  result: Array<IListItem>
  success: boolean
  errors: Array<any>
  messages: Array<any>
  result_info: IResult_info
}
export const getAllType = (): Array<IEnumSelect> => {
  const result: Array<IEnumSelect> = []
  let index = 0
  for (const key in RecordType) {
    const element = RecordType[key]
    result.push({
      uid: index,
      label: key,
      value: element,
      key: key
    })
    index++
  }
  return result
}

export const item2param = (item?: IListItem): IParam => {
  return {
    record_id: item?.id ?? undefined,
    type: item?.type ?? RecordType.A,
    name: item?.name ?? '',
    content: item?.content ?? '',
    ttl: item?.ttl ?? 1,
    proxied: item?.proxied ?? true
  }
}
export const list_svc = (): Promise<IResponse<IListResult>> => {
  return request.get(`${base_url}`, { timeout: 15 * 1000 })
}
export const detail_svc = (recordId: string): Promise<IResponse<any>> => {
  return request.get(`${base_url}/${recordId}`)
}
export const add_svc = (data: IParam): Promise<IResponse<any>> => {
  return request.put(`${base_url}`, data)
}
export const edit_svc = (data: IParam): Promise<IResponse<any>> => {
  return request.patch(`${base_url}`, data)
}

export const delete_svc = (recordId: string): Promise<IResponse<any>> => {
  return request.delete(`${base_url}/${recordId}`)
}
