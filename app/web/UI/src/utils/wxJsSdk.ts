// 假设你已经从后端服务器获取到了以下数据
var appId = 'your-app-id'
var timestamp = 'your-timestamp'
var nonceStr = 'your-nonce-str'
var signature = 'your-signature'

// 1. 配置微信JS-SDK
wx.config({
  debug: false, // 开启调试模式
  appId: appId,
  timestamp: timestamp,
  nonceStr: nonceStr,
  signature: signature,
  jsApiList: [
    'openLocation',
    'onMenuShareTimeline',
    'onMenuShareAppMessage',
    'chooseImage', // 示例API
    'previewImage' // 示例API
  ]
})

// 2. 注册成功和失败的回调
wx.ready(function () {
  // 微信JS-SDK配置成功
  console.log('微信JS-SDK配置成功')

  // 在这里可以调用微信JS-SDK提供的API
  // 例如分享到朋友圈
  wx.onMenuShareTimeline({
    title: '分享标题', // 分享标题
    link: window.location.href, // 分享链接
    imgUrl: 'http://www.example.com/image.png', // 分享图标
    success: function () {
      alert('分享成功')
    },
    cancel: function () {
      alert('分享已取消')
    },
    fail: function (err) {
      console.log(err)
    }
  })
})

wx.error(function (res) {
  // 微信JS-SDK配置失败
  console.log('微信JS-SDK配置失败', res)
})

wx.openLocation({
  latitude: tlat, //目的地latitude
  longitude: tlng, //目的地longitude
  name: '江北机场T3航站楼',
  address: '重庆 江北机场T3航站楼',
  scale: 15 //地图缩放大小，可根据情况具体调整
})
