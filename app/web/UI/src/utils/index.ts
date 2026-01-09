import { getStoreInstance } from 'co6co'

export const isDebug = Boolean(Number(import.meta.env.VITE_IS_DEBUG))
export const setBaseUrl = () => {
  const store = getStoreInstance()
  //const baseUrl = import.meta.env.VITE_BASE_URL
  // 使用类型断言解决TypeScript类型问题
  const _window = window as any
  const baseUrl = _window._env_.API_URL 
  //console.info(_window._env_)
  //console.info(baseUrl)
  store.setBaseUrl(baseUrl)
}

export const getPublicURL = (path: string) => {
  const root = import.meta.env.VITE_UI_PATH
  const url = new URL(`${root}${path}`, import.meta.url).href
  return url
}
