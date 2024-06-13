import Cookie from 'js-cookie'
import { Storage } from './storage'

const authonKey = 'Authorization'
const storage = new Storage()
export function setToken(token: any, secounds = 7200) {
  storage.set(authonKey, token, secounds)
  //localStorage.setItem(authonKey, token);
  return Cookie.set(authonKey, token)
}
export function getToken() {
  let token = storage.get(authonKey)
  if (!token) token = getCookie(authonKey)
  return token
}
export function removeToken() {
  storage.remove(authonKey)
  return Cookie.remove(authonKey)
}
export function getCookie(key: string) {
  const data = Cookie.get(key)
  if (data) return data
  else return ''
}

export function setCookie(key: string, value: string) {
  const val = Cookie.get(key)
  return Cookie.set(key, val || value)
}
export function removeCookie(key: string) {
  return Cookie.remove(key)
}
