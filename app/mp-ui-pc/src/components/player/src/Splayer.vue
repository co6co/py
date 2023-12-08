<template>
    <div ref="videoContainer" class="max-w-full">
      <video  
        ref="videoPlayer"
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

onUnmounted(() => {
  player && player.destroy();
}); 
 
const initApp = () => {  
  if (shaka.Player.isBrowserSupported()) {
    initPlayer();
  } else { 
    console.error("Browser not supported!");
  } 
};

const videoPlayer = ref();
const videoContainer = ref();
let player = {  }
const initPlayer = () => { 
  player = new shaka.Player(videoPlayer.value); 

  const ui = new shaka.ui.Overlay(
    player,
    videoContainer.value,
    videoPlayer.value
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
    await player.load(props.option.url);
  } catch (e) {
    onError(e);
  }
};

const onError = (error:any) => {
  console.error("Error code", error.code, "object", error);
};

const play = () => {
  console.info(player) 
 
  videoPlayer.value && videoPlayer.value.play();alert("player"),console.info( videoPlayer.value)}
const pause = () => videoPlayer.value && videoPlayer.value.pause();

defineExpose({ play, pause });
</script>



<style scoped>
.max-w-full {
  max-width: 100%;
}

.full {
  width: 100%;
  height: 100%;
}
</style> 
<!--
<shaka-player
  class="video"
  :src="src"
  :poster="poster"
  autoplay
></shaka-player>
-->