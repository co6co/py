
import  {jssdk as  jsApi} from '@/api'
import { Point, Storage } from 'co6co'
//import { showFailToast } from 'vant'
interface IOpenLocationParam extends Point {
  /** 缩放级别 1~28，默认为最大*/
  scale?: number
  /**地址 */
  address: string
  /**地址描述 */
  info: string
}
interface IShareInfo {
  title: string
  desc: string
  link: string
  imgUrl?: string
}
export const useJsApi = (jsApiList?: Array<String>,showFailToast:(msg:string)=>void=(msg)=>console.info(msg),isDebug:boolean =false) => {
  jsApi.getSignature(window.location.href).then((res) => {
    console.warn('配置微信JSAPI', res)
    /*
      const index = window.location.hash.indexOf('?')
      let url = `${window.location.origin}${window.location.pathname}`
      if (index > 0) {
        const query = '?' + window.location.hash.split('?')[1]
        url = `${window.location.origin}${window.location.pathname}${query}`
      }
    */
    wx.config({
      debug: isDebug,
      appId: res.data.appId,
      timestamp: res.data.timestamp,
      nonceStr: res.data.nonceStr,
      signature: res.data.signature,
      jsApiList: jsApiList
        ? jsApiList
        : [
            'onMenuShareAppMessage',
            'getLocation',
            'openLocation',
            'onMenuShareTimeline',
            'chooseWXPay',
            'showOptionMenu',
            'updateAppMessageShareData',
            'hideMenuItems',
            'showMenuItems',
            'onMenuShareTimeline'
          ]
    })

    wx.error((res) => {
      console.info('JS-SDK初始化失败:', res)
      showFailToast(`JS-SDK初始化失败:${res}`)
    })
  })

  /**
   * 打开为位置
   * @param item
   * @param success
   * @param fail
   */
  const onOpenLocation = (
    item: IOpenLocationParam,
    success?: (res: any) => void,
    fail?: (res) => void
  ) => {
    wx.ready(function () {
      wx.openLocation({
        latitude: item.lat, // 纬度，范围为90 ~ -90
        longitude: item.lng, // 经度，范围为180 ~ -180
        scale: item.scale ? item.scale : 28, // 缩放级别，整形值，范围从1~28。默认为最大
        info: item.info, // 地址的描述
        address: item.address, // 地址详情说明
        success: function (res) {
          success ? success(res) : console.log('地图已打开', res)
        },
        fail: function (res) {
          fail ? fail(res) : showFailToast(`打开地图失败:${res}`),
            console.info(`打开地图失败:${res}`)
        }
      })
    })
  }
  /**
   * 打开分享
   */
  const shareToWeChat = (info: IShareInfo, success?: (res: any) => void, fail?: (res) => void) => {
    wx.ready(() => {
      wx.onMenuShareAppMessage({
        ...{
          success: function (res) {
            success ? success(res) : console.log('分享成功', res)
          },
          fail: function (res) {
            fail ? fail(res) : showFailToast(`分享失败:${res}`), console.info(`分享失败:${res}`)
          }
        },
        ...info
      })
    })
  }

  /**
   * 获取位置信息
   */
  const getLocation = (
    cacheTimeout: number = 60,
    success?: (point: Point) => void,
    fail?: (err) => void
  ) => {
    console.info('获取位置信息...')
    const store = new Storage()
    const getLocationKey = 'locationKey30'
    const point: Point | null = store.get(getLocationKey)
    if (point) {
      success ? success(point) : console.info('从缓存中获取当前位置信息', point)
    } else {
      wx.ready(() => {
        wx.getLocation({
          type: 'wgs84', // 返回可以用于wx.openLocation的经纬度
          success: function (res) {
            const point: Point = { lng: res.longitude, lat: res.latitude }
            store.set(getLocationKey, point, cacheTimeout)
            success ? success(point) : console.info('获取位置信息：', point)
          },
          fail: function (err) {
            fail ? fail(err) : showFailToast(`获取位置失败:${JSON.stringify(err)}`),
              console.info(`获取位置失败:${err}`)
          }
        })
      })
    }
  }
  return { onOpenLocation, shareToWeChat, getLocation }
}
