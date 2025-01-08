import { createAxiosInstance, HttpContentType, useRequestToken } from 'co6co'
const base_URL = '/api/files'
export const file_content_svc = (path: string) => {
  const request = createAxiosInstance(undefined, 5000, HttpContentType.json, HttpContentType.text)
  request.interceptors.request.use(useRequestToken)
  return request.post(`${base_URL}/file`, { path: path })
}
