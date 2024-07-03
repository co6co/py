/* eslint-disable camelcase */
import axios, { type AxiosResponse } from 'axios'
import { type IPageResponse, type IResponse } from 'co6co'
import request from '../../utils/request'
import { download_svc } from 'co6co-right'

const BASE_URL = '/api/biz/process'
export const get_status_svc = (): Promise<IPageResponse> => {
  return request.get(`${BASE_URL}/getStatus`)
}

export const queryList_svc = (data: any): Promise<IPageResponse> => {
  return request.post(`${BASE_URL}/list`, data, { data })
}
export const audit_svc = (data: any): Promise<IResponse> => {
  return request.post(`${BASE_URL}/audit`, data)
}
export const position_svc = (data: any): Promise<IResponse> => {
  return request.post(`${BASE_URL}/position`, data)
}
export const one_svc = (id: number): Promise<IResponse> => {
  return request.get(`${BASE_URL}/one/${id}`)
}

export const start_download_task = (data: any): Promise<IResponse> => {
  return request.post(`${BASE_URL}/startDownloadTask`, data)
}

//单独下载
export const download_one_svc = (
  id: number,
  data: { boatName: string; vioName: string },
  bck: () => void
) => {
  download_svc(`${BASE_URL}/download/${id}`, `${data.boatName}_${data.vioName}_${id}.zip`, bck)
}

export const get_log_content_svc = (id: number): Promise<AxiosResponse> => {
  return request.post(`${BASE_URL}/auditLog/${id}`, null, {
    responseEncoding: 'utf-8',
    responseType: 'text'
  })
}
