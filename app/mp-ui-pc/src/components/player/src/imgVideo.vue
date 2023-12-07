<template >
    <el-row>
      <el-col :span="24" style="height:25rem;">   
        <!--<el-image  :key="index" v-show="item.type==1"  v-for="(item,index) in viewOption" :src="item.url" style="width: 100%; height: 100%" ></el-image>
        <player   :key="index" v-show="item.type==0"  v-for="(item,index) in viewOption" :option="item"></player>--> 
        <component :key="currentName" :is="compoents[currentName]" :option="current_options"></component>
       </el-col>
      <el-col :span="24">
        <div style="overflow: auto;">
          <ul  style="height:5rem;" >
            <!--{position:(item.type==1?'inherit':'relative')}--> 
            <li @click="onShow(item,index)" :key="index" v-for="(item,index) in imageOptions"   >  
             <a><el-image :key="index" :src="item.poster" :title="item.name"/> </a>   
            </li>
            <li @click="onShow(item,index)" :key="index" v-for="(item,index) in videoOptions"  style="position:relative" >  
              <a> <el-image :src="item.poster" :title="item.name"/><CaretRight /> </a>  
            </li> 
          </ul>  
        </div>
      </el-col> 
    </el-row>
</template>
<script setup lang="ts">
import { watch, PropType,reactive, ref , markRaw,defineAsyncComponent,computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import 'vue3-video-play/dist/style.css'
import { resourceOption } from './types'
import { ElImage } from 'element-plus';
const props = defineProps({ 
  viewOption:{
    type:Array<resourceOption>,
      required:true 
  } 
}) 
const compoents = reactive({
  Image: markRaw(defineAsyncComponent(() => import('../../../components/player/src/Image.vue'))),
  Video: markRaw(defineAsyncComponent(() => import('../../../components/player/src/Player.vue'))) 
}) 
const imageOptions= computed  (()=>{
    return props.viewOption.filter(m=>m.type==1)
})
 
const videoOptions= computed  (()=>{
    return props.viewOption.filter(m=>m.type==0)
})
const currentName = ref<"Image"|"Video">('Image')  
const current_options = ref<resourceOption>({ url:"", name:"string", type:1,poster:""} ) 
  const onShow=(option:resourceOption, key:number)=>{ 
  if (option.type==0){
    currentName.value="Video" 
  }
  else currentName.value="Image"
  current_options.value=option
}
</script> 
<style scoped lang="less">
ul{
  width: 833px;
  li {  
        cursor:pointer; border: 1px #ccc ;overflow: hidden;list-style: none;  
        float: left; min-width: 100px; text-align: center; margin: 4px; 
       a {
        .el-image{width: 200px;height: 106px;}
        svg{position: absolute; left: 40%; top: 38%; width: 20%;}
      } 
   } 
}
</style>