import request from '../../utils/request'
import { type IResponse } from 'co6co'
const BASE_URL = '/api/app/config'

export interface ClientConfig {
  batchAudit: boolean //批量审核提交
}

export const get_config = (): Promise<IResponse<ClientConfig>> => {
  return request.get(`${BASE_URL}`)
}
