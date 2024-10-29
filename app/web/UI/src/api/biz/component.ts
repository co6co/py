import request from '../../utils/request'

import type { IResponse } from 'co6co'
const base_URL = '/api/biz/components'
export const get_component_code = (code: string): Promise<IResponse<string>> => {
  return request.get(`${base_URL}/${code}`)
}
