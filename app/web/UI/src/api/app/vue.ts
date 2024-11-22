export const vuePath = import.meta.env.VITE_UI_PATH
import { createAxios } from 'co6co'
export const createVueRequest = () => {
  return createAxios({ baseURL: vuePath })
}
export const get_history_svc = () => {
  return createVueRequest().get('/history.txt')
}
