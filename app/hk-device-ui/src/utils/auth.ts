import Cookie from 'js-cookie'
const TokenKey = 'vue_token'

export function getToken() {
  return Cookie.get(TokenKey)
}

export function setToken(newToken:any) {
  const token = Cookie.get('token');
  return Cookie.set(TokenKey, token || newToken)
}

export function removeToken() {
  return Cookie.remove(TokenKey)
}