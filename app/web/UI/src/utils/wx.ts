import { showNotify } from 'vant'
import { randomString } from 'co6co'
import { useRouter, useRoute } from 'vue-router'

const appid = import.meta.env.VITE_WX_appid
const debug = Boolean(Number(import.meta.env.VITE_IS_DEBUG))

export enum wxSnsApiCategory {
  snsapi_base = 'snsapi_base',
  snsapi_userinfo = 'snsapi_userinfo'
}

/** 微信服务器地址 */
export const RedirectWxService = (
  redirect_uri: string,
  scope: wxSnsApiCategory,
  stateCode: string
) => {
  if (debug) return `${redirect_uri}&code=123456&scope=${scope}&state=${stateCode}#wechat_redirect`
  else {
    //console.info('编码前：', redirect_uri)
    redirect_uri = encodeURIComponent(redirect_uri) // 对除了 字母数字和-, _, ., ~ 之外的 所有字符编码
    //redirect_uri = encodeURI(redirect_uri)
    //console.info('编码后：', redirect_uri)
    let redirectUrl = `https://open.weixin.qq.com/connect/oauth2/authorize?appid=${appid}&redirect_uri=${redirect_uri}&response_type=code&scope=${scope}&state=${stateCode}#wechat_redirect`
    return redirectUrl
  }
}
/**
 * 获取当前URL
 *
 * @returns  url 全路径
 */
export const getCurrentUrl = () => {
  return document.location.toString()
}
/**
 * 获取URL path
 *
 * http://localhost:5173/xd/home.html#/userInfo?tikite=123
 * 获得的值为：/xd/home.html#/userInfo
 * 没有参数
 * @returns /path
 */
export const getPathAndQuery = () => {
  // 获取路径名
  const pathName = window.location.pathname // /xd/home.html
  // 获取查询字符串
  const search = window.location.hash //#/xxx?a=b&c=d
  if (debug) return getCurrentUrl()
  return `${pathName}${search}`
}
export const getUrlPath = () => {
  let url = getCurrentUrl()
  let arrUrl = url.split('//')
  let start = arrUrl[1].indexOf('/')
  let relUrl = arrUrl[1].substring(start) //stop省略，截取从start开始到结尾的所有字符
  if (relUrl.indexOf('?') != -1) {
    relUrl = relUrl.split('?')[0]
  }
  return relUrl
}
/** 通过微信服务器地址跳转会 本地服务器地址 */
const redirectUrl = () => {
  showNotify({ type: 'warning', message: `跳转...` })
  const vite_config_redirect_uri = import.meta.env.VITE_WX_redirect_uri
  //console.info('vite_config_redirect_uri：', vite_config_redirect_uri)
  //const scope = wxSnsApiCategory.snsapi_userinfo
  const scope = wxSnsApiCategory.snsapi_userinfo
  // 需要对 # 进行编码
  const backUrl = encodeURIComponent(`${getPathAndQuery()}`)
  let url = debug
    ? `${vite_config_redirect_uri}?backurl=${backUrl}`
    : `${window.location.protocol}//${window.location.host}${vite_config_redirect_uri}?backurl=${backUrl}`
  const redirectUrl = RedirectWxService(url, scope, `${randomString(15)}${scope}`)
  window.location.href = redirectUrl
}
export { redirectUrl }
