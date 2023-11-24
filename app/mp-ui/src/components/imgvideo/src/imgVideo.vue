<template >
    <el-row>
      <el-col :span="24" style="height:25rem;">  
        <el-image  v-if="showContent==0" style="width: 100%; height: 100%" :src="picUrl"/>  
        <player    v-if="showContent==1" :option="videoOptions[0]"></player>
        <el-image  v-if="showContent==2" style="width: 100%; height: 100%" :src="picUrl2"   /> 
        <player    v-if="showContent==3" :option="videoOptions[1]"></player>
      </el-col>
      <el-col :span="24">
        <div style="overflow: auto;">
          <ul  style="height:5rem;" >
            <li @click="showContent=0" > <a> <el-image :src="picUrl" title="显示原始图片"/>  </a></li>
            <li @click="showContent=1"  style="position: relative;"><a> <el-image :src="videoOptions[0].poster" title="原始视频"/> <CaretRight /> </a></li>
            <li @click="showContent=2" > <a><el-image :src="picUrl2" title="标注后图片"/></a></li>
            <li @click="showContent=3"  style="position: relative;"><a><el-image :src="videoOptions[1].poster" title="标注后视频"/><CaretRight/> </a></li>
          </ul> 
        </div>
      </el-col>
    </el-row>
</template>
<script setup lang="ts">
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import 'vue3-video-play/dist/style.css'
import {Player,types } from '../../player'
const props = defineProps({ 
  picUrl: {
    type:  String,
    required: false
  },
  picUrl2: {
    type:  String,
    required: false
  }, 
  videoOptions:{ 
    type:Array<types.videoOption>  ,
    required: true
  }
})    
let showContent=ref<number>(0)  
</script> 
<style scoped lang="less">
ul{
  width: 833px;
  li { cursor:pointer; border: 1px #ccc ;float: left; min-width: 100px; text-align: center; margin: 4px;
       overflow: hidden;height: 100%;list-style: none;  
       a {
        .el-image{width: 200px;height: 106px;}
        svg{position: absolute; left: 40%; top: 38%; width: 20%;}
      } 
    } 
}
</style>