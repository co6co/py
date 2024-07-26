export const isDebug = Boolean(Number(import.meta.env.VITE_IS_DEBUG))
import { getStoreInstance } from 'co6co'
export const setBaseUrl = () => {
  const store = getStoreInstance()
  const baseUrl = import.meta.env.VITE_BASE_URL
  console.info(baseUrl)
  store.setBaseUrl(baseUrl)
}

export const getPublicURL = (path: string) => {
  const root = import.meta.env.VITE_UI_PATH
  const url = new URL(`${root}${path}`, import.meta.url).href
  return url
}
