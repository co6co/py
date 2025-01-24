[示例](https://www.weekweekup.cn/contribute/detail?i=26)

# 文件/文件夹说明
文件 | 说明
:-: | :-: 
TcPlayer-2.3.2_hava.js | 基于TcPlayer-2.3.2改造后倍速\防盗录\自定义加密的tcplayer文件,可直接使用,仅用于参考
hls.0.12.4_hava.js | 基于hls.0.12.4.js改造后带有hls加密hls.js文件,不可直接使用;使用方式,在文件中定位"解密操作"字样,加入自定义的解密方式,将解密后的m3u8索引字符串赋值给u
# 使用说明
## 增加参数&说明
参数 | 类型 | 默认值 | 参数说明
:-: | :-: | :-: | :-:
encryptHls | String | 无 | 表示调用的用于解析视频的hls文件,设置该参数表示开启自定义解析(可加入自定义加密)
rates | Array | [2, 1.75, 1.5, 1.25, 1, 0.75, 0.5] |　倍速数组
curRate | Number | 1 | 默认倍速
appear_text | String | 无 | 防录屏文字,无则表示不出现防录屏文字
appear_time | Number | 10 | 防录屏文字出现是时长最大值
disappear_time | Number | 100 | 防录屏文字消失的时长最大值
appear_color | Array | ["#fff", "#000"] | 防录屏文字出现时的颜色
appear_fontsize_min | Number | 12 | 防录屏文字出现时字体的最小值
appear_fontsize_max | Number | 22 | 防录屏文字出现时字体的最大值
subtitle_display | Boolean | false | 是否显示字幕
subtitle_srt | String | 无 | srt字幕文件地址,不设置则不在底部栏显示字幕按钮
subtitle_fontsize | String | "16px" | 字幕字体大小
subtitle_fullscreen_fontsize | String | "32px" | 全屏时字幕字体大小

## 增加方法&说明
方法 | 参数 | 返回值 | 说明 | 示例
:-: | :-: | :-: | :-: | :-:
currentRate() | 无 | {int} | 获取当前的倍速 | player.currentRate()

## 使用示例

```javascript
var player = new TcPlayer("id_test_video", {
	"m3u8": "http://2157.liveplay.myqcloud.com/2157_358535a.m3u8",
	"autoplay" : true,
	"poster" : "http://www.test.com/myimage.jpg",
	"width" :  "480",
	"height" : "320",
	"x5_player": true,
	"systemFullscreen": true,
	"x5_type": "h5",
	"x5_fullscreen": true,
	"x5_orientation": 2,
	"encryptHls": "./hls.0.12.4_hava.js",
	"rates": [2, 1.5, 1.0, 0.5],
	"curRateIndex": 2,
	"appear_text": "大洼X",
	"appear_time": 10,
	"disappear_time": 100,
	"appear_color": ["#fff", "#F4F4F4"],
	"appear_fontsize_min": 12,
	"appear_fontsize_max": 22,
	"subtitle_display": true,
	"subtitle_srt": "字幕文件",
	"subtitle_fontsize": "28px",
	"subtitle_fullscreen_fontsize": "56px",
	
});
```

# 教程
- [tcplayer源码改造第一弹 -> 自定义hls加密播放器](https://blog.csdn.net/z13192905903/article/details/102862664)
- [tcplayer源码改造第二弹 -> 加入倍速播放](https://blog.csdn.net/z13192905903/article/details/102862664)
- [tcplayer源码改造第三弹 -> 防盗录](https://blog.csdn.net/z13192905903/article/details/103366173)
- [tcplayer 源码改造第四弹 -> 字幕(srt)](https://blog.csdn.net/z13192905903/article/details/103424010)
- [tcplayer源码改造第五弹 -> 兼容sarafi/遨游](https://blog.csdn.net/z13192905903/article/details/103851286)
