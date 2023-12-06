<template >
    <el-row>
      <el-col :span="24" style="height:25rem;">   
         <el-image  :key="index" v-show="item.type==1"  v-for="(item,index) in viewOption" :src="item.url" style="width: 100%; height: 100%" ></el-image>
       <!--<player   :key="index" v-show="item.type==0"  v-for="(item,index) in viewOption" :option="item"></player>
        <component :is="currentTemplate"></component> -->
       </el-col>
      <el-col :span="24">
        <div style="overflow: auto;">
          <ul  style="height:5rem;" >
            <li @click="showContent=index" :key="index" v-for="(item,index) in viewOption" :style="{position:(item.type==1?'inherit':'relative')}" >  
              <a v-if="item.type==1"> <el-image :src="item.url" :title="item.name"/>  </a>
              <a v-else> <el-image :src="item.poster" title="原始视频"/> <CaretRight /> </a>  
            </li>
          </ul> 
           
        </div>
      </el-col>
    </el-row>
</template>
<script setup lang="ts">
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import 'vue3-video-play/dist/style.css'
import { videoOption,imageOption } from './types'  
import {Player} from '../../../components/player' 
const props = defineProps({ 
  viewOption:{
    type:Array<imageOption|videoOption>,
      required:true 
  } 
})    
 
let showContent=ref<number>(0)  
const onShow=(key:number)=>{
  showContent
}
const currentTemplate=ref()
</script> 
<style scoped lang="less">
ul{
  width: 833px;
  li { cursor:pointer; border: 1px #ccc ;float: left; min-width: 100px; text-align: center; margin: 4px;
       overflow: hidden;list-style: none;  
       a {
        .el-image{width: 200px;height: 106px;}
        svg{position: absolute; left: 40%; top: 38%; width: 20%;}
      } 
   } 
}
</style>