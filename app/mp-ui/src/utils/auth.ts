import Cookie from 'js-cookie'
import { Storage } from '../store/Storage'
const authonKey = 'Authorization3'

const storage = new Storage()
export function setToken(token: any, secounds: number = 7200) {
  try {
    storage.set(authonKey, token, secounds)
  } catch (e) {
    console.warn('setToken Error:', e)
  } 
  //localStorage.setItem(authonKey, token);
  //return Cookie.set(authonKey, token);
}
export function getToken() {
  let token = storage.get(authonKey)
  //if (!token) token = getCookie(authonKey)
  return token
}
export function removeToken() {
  storage.remove(authonKey)
  return Cookie.remove(authonKey)
}
export function getCookie(key: string) {
  let data = Cookie.get(key)
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
