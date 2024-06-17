/* eslint-disable no-console */
/// <reference types="../../typings/global" />
import JSONbig from 'json-bigint'
import dayjs from 'dayjs'
import type { ITreeSelect } from '@co6co/constants'

export const str2Obj = (str: string) => {
  return JSONbig.parse(str)
}

export const isDebug = Boolean(Number(import.meta.env.VITE_IS_DEBUG))
export const sleep = (time: number) => {
  return new Promise((resolve) => setTimeout(resolve, time))
}
/// 创建开始结束时间
export const createStateEndDatetime = (type: number, beforeHour: number) => {
  let endDate: Date | number = new Date()
  let startDate = new Date()
  let times = -1
  switch (type) {
    case 0:
      endDate = new Date()
      times = endDate.getTime() - beforeHour * 3600 * 1000
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
    dayjs(endDate).format('YYYY-MM-DD HH:mm:ss'),
  ]
}

//生成随机字符串
export const randomString = (
  len: number,
  chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
) => {
  let result = ''
  for (let i = len; i > 0; --i)
    result += chars[Math.floor(Math.random() * chars.length)]
  return result
}
//获取URL 参数
export const getQueryVariable = (key: string) => {
  try {
    //var query = window.location.search.substring(1);
    const query = window.location.href.slice(
      Math.max(0, window.location.href.indexOf('?') + 1)
    )
    const vars = query.split('&')
    for (const var_ of vars) {
      const pair = var_.split('=')
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

export const toggleFullScreen = (
  elem: HTMLElement | any,
  fullScreen = true
) => {
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
  if (len) while (hex.length < len) hex = `0${hex}`
  return hex
}

type Collection<T> = (a: Array<T>, b: Array<T>) => Array<T>
// 交集
export const intersect = (array1: [], array2: []) =>
  array1.filter((x) => array2.includes(x))

// 差集
export const minus: Collection<number | string> = (
  array1: Array<number | string>,
  array2: Array<number | string>
) => array1.filter((x) => !array2.includes(x))
// 补集
export const complement = (array1: [], array2: []) => {
  array1
    .filter((v) => {
      return !array2.includes(v)
    })
    .concat(
      array2.filter((v) => {
        return !array1.includes(v)
      })
    )
}
// 并集
export const unionSet = (array1: [], array2: []) => {
  return array1.concat(
    array2.filter((v) => {
      return !array1.includes(v)
    })
  )
}
//key down demo
export const onKeyDown = (e: KeyboardEvent) => {
  console.info('key', e.key)
  if (e.ctrlKey) {
    if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
      //let current = table_module.query.pageIndex.valueOf();
      //let v = e.key == 'ArrowRight' || e.key == 'd' ? current + 1 : current - 1;
      //onPageChange(v);
    }
    if (['ArrowUp', 'ArrowDown'].includes(e.key)) {
      //let current = currentTableItemIndex.value;
      //if (!current) current = 0;
      //let v = e.key == 'ArrowDown' || e.key == 's' ? current + 1 : current - 1;
      //if (0 <= v && v < tableInstance._value.data.length) {
      //	setTableSelectItem(v);
      //} else {
      //	if (v < 0) ElMessage.error('已经是第一条了');
      //	else if (v >= tableInstance._value.data.length) ElMessage.error('已经是最后一条了');
      //}
    }
  }
  //process_view.value.keyDown(e)
  e.stopPropagation()
}

/**
 * 遍历 Tree
 * @param tree
 * @param func  return true 退出循环
 */

export const traverseTreeData = (
  tree: Array<ITreeSelect>,
  func: (data: ITreeSelect) => void | boolean
) => {
  tree.forEach((data) => {
    data.children && traverseTreeData(data.children, func) // 遍历子树
    const result = func(data)
    if (result) return
  })
}
