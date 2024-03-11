import JSONbig from 'json-bigint'
import { dayjs } from 'element-plus'

import config from '../../package.json'
export const pkg = config

export const str2Obj = (str: string) => {
  return JSONbig.parse(str)
}
export const objectForEach = (
  obj: { [key: string]: any },
  bck: (value: any, key?: string, index?: number) => void
) => {
  let keys = Object.keys(obj)
  for (let i = 0; i < keys.length; i++) {
    bck(obj[keys[i]], keys[i], i)
  }
}

export const sleep = (time: number) => {
  return new Promise((resolve) => setTimeout(resolve, time))
}

/// 创建开始结束时间
export const createStateEndDatetime = (type: number, beforeHour: number) => {
  let endDate = null
  let startDate = null
  switch (type) {
    case 0:
      endDate = new Date()
      // eslint-disable-next-line no-case-declarations
      const times = endDate.getTime() - beforeHour * 3600 * 1000
      startDate = new Date(times)
      break
    case 1:
      startDate = new Date(dayjs(new Date()).format('YYYY/MM/DD'))
      endDate = startDate.getTime() + 24 * 3600 * 1000 - 1000
      break
    default:
      startDate = new Date(dayjs(new Date()).format('YYYY/MM/DD'))
      endDate = startDate.getTime() + 24 * 3600 * 1000 - 1000
      break
  }
  return [
    dayjs(startDate).format('YYYY-MM-DD HH:mm:ss'),
    dayjs(endDate).format('YYYY-MM-DD HH:mm:ss')
  ]
}

//生成随机字符串
export const randomString = (
  len: number,
  chars: string = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
) => {
  let result = ''
  for (let i = len; i > 0; --i) result += chars[Math.floor(Math.random() * chars.length)]
  return result
}
//获取URL 参数
export const getQueryVariable = (key: string) => {
  try {
    //var query = window.location.search.substring(1);
    const query = window.location.href.substring(window.location.href.indexOf('?') + 1)
    const vars = query.split('&')
    for (let i = 0; i < vars.length; i++) {
      const pair = vars[i].split('=')
      if (pair[0] == key) {
        return pair[1]
      }
    }
    return null
  } catch (e) {
    console.error('queryVariable Error:', e)
  }
  return null
}

export const toggleFullScreen = (elem: HTMLElement | any, fullScreen: boolean = true) => {
  if (!elem) elem = document.documentElement
  if (fullScreen) {
    if (elem.requestFullscreen) {
      console.info('1')
      elem.requestFullscreen()
    } else if (elem.mozRequestFullScreen) {
      console.info('2')
      elem.mozRequestFullScreen()
    } else if (elem.webkitRequestFullscreen) {
      console.info('3')
      elem.webkitRequestFullscreen()
    } else if (elem.msRequestFullscreen) {
      console.info('4')
      elem.msRequestFullscreen()
    }
  } else {
    elem = document
    if (elem.exitFullscreen) {
      elem.exitFullscreen()
    } else if (elem.mozCancelFullScreen) {
      // Firefox
      elem.mozCancelFullScreen()
    } else if (elem.webkitExitFullscreen) {
      // Chrome, Safari and Opera
      elem.webkitExitFullscreen()
    } else if (elem.msExitFullscreen) {
      // Internet Explorer and Edge
      elem.msExitFullscreen()
    }
  }
}

//10进制转16进制补0
export const number2hex = (dec: number, len: number) => {
  let hex = ''
  while (dec) {
    const last = dec & 15
    hex = String.fromCharCode((last > 9 ? 55 : 48) + last) + hex
    dec >>= 4
  }
  if (len) while (hex.length < len) hex = '0' + hex
  return hex
}
