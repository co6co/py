<template>
  <el-row>
    <el-col style="height: 25rem"> 
      <el-image v-if="showContent == 0" style="width: 100%; height: 100%" :src="imageOption[0].url" />
      <html-player v-if="showContent == 1" :option="videoOptions[0]"></html-player>
      <el-image v-if="showContent == 2" style="width: 100%; height: 100%" :src="imageOption[1].url" />
      <html-player v-if="showContent == 3" :option="videoOptions[1]"></html-player>
    </el-col>
    <el-col > 
        <el-scrollbar>
        <ul style="height:5rem"> 
          <li @click="showContent = 0">
            <a> <el-image :src="imageOption[0].thumbnail" title="显示原始图片" /> </a>
          </li>
          <li @click="showContent = 1" style="position: relative">
            <a> <el-image :src="videoOptions[0].poster" title="原始视频" /> <CaretRight /> </a>
          </li>
          <!-- 后台不会生成移除
          <li @click="showContent = 2">
            <a><el-image :src="imageOption[1].thumbnail" title="标注后图片" /></a>
          </li>
          -->
          <li @click="showContent = 3" style="position: relative">
            <a><el-image :src="videoOptions[1].poster" title="标注后视频" /><CaretRight /> </a>
          </li> 
        </ul>
      </el-scrollbar>
       
    </el-col>
  </el-row>
</template>
<script setup lang="ts">
import {
  watch,
  type PropType,
  reactive,
  ref,
  computed,
  onMounted,
  onBeforeUnmount,
  nextTick
} from 'vue'

import { htmlPlayer, types } from '../../player'
const props = defineProps({ 
  imageOption: {
    type: Array<types.imageOption>,
    required: true
  },
  videoOptions: {
    type: Array<types.videoOption>,
    required: true
  }
})
let showContent = ref<number>(0)
</script>
<style scoped lang="less">
ul {
  width: 833px;
  li {
    cursor: pointer;
    border: 1px #ccc;
    float: left;
    min-width: 100px;
    text-align: center;
    margin: 4px;
    overflow: hidden;
    height: 100%;
    list-style: none;
    a {
      .el-image {
        width: 200px;
        height: 106px;
      }
      svg {
        position: absolute;
        left: 40%;
        top: 38%;
        width: 20%;
      }
    }
  }
}
</style>
