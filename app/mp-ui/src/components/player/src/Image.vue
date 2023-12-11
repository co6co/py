<template>   
{{ result }}
    <el-image :src="result" style="width: 100%; height: 100%"  :title="option.name" ></el-image>
</template>

<script lang="ts" setup  >
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';   
import { imageOption } from './types'
import  * as res_api from '../../../api'; 
const props = defineProps({
 option: {
   type:  Object as PropType<imageOption>,
   required: true
}}) 
 const result=ref("") 
watch(()=>props.option,(n,o)=>{
  res_api.request_resource_svc(props.option.url) .then(response => result.value=response)
},{immediate:true})
</script> 