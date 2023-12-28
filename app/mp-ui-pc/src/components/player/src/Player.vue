<template> 
  <videoPlay v-if="option.url"  v-bind="playerOption" :poster='option.poster'></videoPlay>  
  <el-empty v-else description="未加载数据" />
</template>

<script lang="ts" setup  >
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import 'vue3-video-play/dist/style.css'
import { videoPlay} from 'vue3-video-play'
import {videoOption } from './types'
const props = defineProps({
 option: {
   type:  Object as PropType<videoOption>,
   required: true
}
}) 
const data=ref("")
const playerOption=reactive({
 width: '100%', //播放器高度O
 height: '100%', //播放器高度
 color: "#409eff", //主题色
 title: '', //视频名称
 src: data.value, //视频源
 muted: false, //静音
 webFullScreen: false,
 speedRate: ["0.75", "1.0", "1.25", "1.5", "2.0"], //播放倍速
 autoPlay: false, //自动播放
 loop: false, //循环播放
 mirror: false, //镜像画面
 ligthOff: false,  //关灯模式
 volume: 0.3,    //默认音量大小
 control: true,  //是否显示控制器
})  
onMounted(()=>{
  playerOption.src=props.option.url 
})
</script> 