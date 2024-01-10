 <template>
    <div ref="playerContainer" class="max-w-full">
      <video  
        ref="playerEle"
        class="full"
        :poster="option.poster"
        :autoplay="autoplay"
        muted
        controls
      ></video>
      <button @click="play">播放</button>
      <button @click="pause">暂停</button>
    </div>
</template>
 
 

<!--https://blog.csdn.net/xiao suom/article/details/1299 89073-->
<script lang="ts" setup>

import { ref, onMounted, watch,onUnmounted, onBeforeUnmount,PropType } from "vue"; 
//import * as shaka from 'shaka-player';
import 'shaka-player/dist/controls.css'; 
 
import 'shaka-player/dist/shaka-player.compiled.js';
import 'shaka-player/dist/shaka-player.compiled.externs.js';
import 'shaka-player/dist/shaka-player.ui.js';
import 'shaka-player/dist/shaka-player.ui.externs.js';
 import {videoOption } from './types'

 
const props = defineProps({
    option: {
        type:  Object as PropType<videoOption>,
        required: true
    },
  autoplay: { type: Boolean, default: false }
});

onMounted(() => {
  initApp();
}); 
const playerContainer = ref(); 
const playerEle = ref();
const player=ref<shaka.Player>()
onUnmounted(() => {
  player.value && player.value.destroy();
}); 
 
const initApp = () => {  
  if (shaka.Player.isBrowserSupported()) {
    initPlayer();
  } else { 
    console.error("Browser not supported!");
  } 
};



const initPlayer = () => {  
  player.value=new shaka.Player(playerEle.value); 
  
  const ui = new shaka.ui.Overlay(
    player.value ,
    playerContainer.value,
    playerEle.value
  ); 
  ui.configure({
    // 自定义控件
    controlPanelElements: [
      "time_and_duration", // 进度
      "spacer",
      "mute", // 静音
      "volume", // 音量
      "fullscreen", // 全屏
      "overflow_menu"
    ],
    
    overflowMenuButtons: [
      "picture_in_picture", // 画中画
      "playback_rate"       // 倍速
    ],
    
    playbackRates: [0.5, 1, 1.5, 2], // 倍速选项
    // 视频进入全屏时设备是否应旋转到横向模式
    forceLandscapeOnFullscreen: false
  }); 
  loadPlayer();
};

const loadPlayer = async () => {
  try {
    if (player.value) await player.value.load(props.option.url);
  } catch (e) {
    onError(e);
  }
};

const onError = (error:any) => {
  console.error("Error code", error.code, "object", error);
};

const play = () => {
  console.info(player)  
  playerEle.value && playerEle.value.play();alert("player"),console.info( playerEle.value)}
const pause = () => playerEle.value && playerEle.value.pause();

defineExpose({ play, pause });
</script>

 