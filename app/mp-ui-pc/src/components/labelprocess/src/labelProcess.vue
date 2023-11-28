<template>
	<div class="container">  
    <el-row :gutter="24">  
      <el-col :span="14"> 
        <img-video :pic-url="pic_url" :pic-url2="pic_url2" :video-options="playerOptions" ></img-video>
      </el-col>
      <el-col :span="10"> 
        <el-row> 
          <el-col :span="24">
            <el-card class="box-card">
              <template #header>
                  <div class="card-header">
                      <span>标签</span> 
                  </div>
              </template>
              <div>
                <div class="container"  style="height: 80%; overflow: auto;margin: 5px;"> 
                  <el-descriptions :column="2">
                    <el-descriptions-item label="船名" >{{ options?.id }}</el-descriptions-item>
                    <el-descriptions-item label="船名" >{{ options?.boatName }}</el-descriptions-item>
                    <el-descriptions-item label="流程状态">
                      <el-tag size="small">{{ options?.flowStatus }}-{{  metaData.getFlowStateName(options?.flowStatus==undefined?-1:options.flowStatus)?.label  }}</el-tag>
                    </el-descriptions-item> 
                  </el-descriptions>
                  <el-descriptions  :column="1">
                    <el-descriptions-item label="违规名称" >
                      <el-tooltip effect="dark" :content="options?.vioName" placement="top">
                        <span class="ellipsis">  {{ options?.vioName }} </span>
                      </el-tooltip>
                    </el-descriptions-item>
                  </el-descriptions>

                <el-descriptions> 
                <el-descriptions-item label="人工审核">
                  <el-tag size="small" :type="metaData.statue2TagType(options?.manualAuditResult)">{{ options?.manualAuditResult }}-{{ metaData.getManualStateName(options?.manualAuditResult==undefined?-1:options.manualAuditResult)?.label}}</el-tag>
                </el-descriptions-item>  
                <el-descriptions-item label="程序审核">
                  <el-tag size="small"  :type="metaData.statue2TagType(options?.programAuditResult)">{{ options?.programAuditResult }}-{{ metaData.getManualStateName(options?.programAuditResult==undefined?-1:options.programAuditResult)?.label }}</el-tag>
                </el-descriptions-item> 
                </el-descriptions> 
                <el-checkbox-group  v-model="labelModel.appendIds" @change="onCheckGroupChange" > 
                    <el-checkbox  v-for="(t,index) in tags" :key="index"  :label="t.id" :name="t.name"> 
                      <el-tooltip :content="t.alias" placement="top" effect="light">
                        {{ t.name }}  
                      </el-tooltip>
                      </el-checkbox> 
                    </el-checkbox-group> 
                </div>   
                <el-button :disabled="!saveLabelButState" type="danger" @click="onSaveLabel">保存</el-button>  
              </div> 
            </el-card>  
          </el-col> 
        </el-row>  
      </el-col>
    </el-row> 
	</div>
</template>
<script setup lang="ts">   
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import {ElImage, ElDescriptions,ElDescriptionsItem,ElTag,ElDivider,ElMessage } from 'element-plus';
import {Check,  Delete, Edit,Message,Notebook,Star,Pointer,WarningFilled,UploadFilled,CaretRight} from '@element-plus/icons-vue'
import { imgVideo ,types} from "../../imgvideo";  

import { mark_list_svc,marked_list_svc,mark_label_svc } from '../../../api/label';
import { request_resource_svc  } from '../../../api';  
import { fa } from 'element-plus/es/locale';
 
const props = defineProps({
  options: {
    type:  Object as PropType<ProcessTableItem>,
    required: false
  },
  metaData:{ 
    type:Object as PropType<ItemAattachData>,
    required: true
  },
})
const emit = defineEmits([ "refesh"])
 /**   图片视频显示 */
let pic_url= computed(()=>{
    if(props.options){ 
      return import.meta.env.VITE_DATA_PATH+props.options.pic1SavePath;
    }
    return "";
}) 
let pic_url2= computed(()=>{
    if(props.options){ 
      return import.meta.env.VITE_DATA_PATH+props.options.annoPic1SavePath; 
    }
    return "";
})
const playerOptions=ref<Array<types.videoOption> >([{url:"",poster:""},{url:"",poster:""}] )
const updatePlayerOptions=(index:number,address?:string)=>{
  if (address){ 
        playerOptions.value[index].url=import.meta.env.VITE_DATA_PATH +address
        request_resource_svc("/api/biz/process/poster?path="+address).then(res=>{ playerOptions.value[index].poster=res}).catch(e=>playerOptions.value[index].poster="" );   
    }else playerOptions.value[index].poster="" 
}
watch(()=>props.options?.videoSavePath, (n?:string,o?:string)=>{ updatePlayerOptions(0,n) })
watch(()=>props.options?.annoVideoSavePath, (n?:string,o?:string)=>{  updatePlayerOptions(1,n) }) 
 /** end 图片视频显示 */ 

const tags=ref<[ {id:number,name:string,alias:string,checked?:boolean}]>()
const res=mark_list_svc().then(res=> {if (res.code==0){
  tags.value=res.data 
}})

 
const loadLabel=(n:number)=>{
    //重置状态 
    tags.value?.map(m=>m.checked=false)
    marked_list_svc(n).then(res=>{
      if(res.code==0){ 
        let labelArray= res.data.map((m: { labelId: number; })=>m.labelId)  
        labelModel.value.dbIds=Array.from(labelArray); 
        tags.value?.filter(m=>labelArray.indexOf( m.id )>-1).map(m=>m.checked=true)
        let checked=tags.value?.filter(m=>m.checked).map(m=>m.id)
        if(checked) labelModel.value.appendIds=checked.concat()
      }
    })

}
const labelModel=ref<{dbIds:Array<number>, appendIds:Array<number> }>({dbIds:[],appendIds:[] })
watch(()=>props.options?.id,(n?:number,o?:number)=>{
  if(n){
    loadLabel(n)
  }
})
if (props.options&& props.options.id)loadLabel(props.options.id)
const array_eq=(arr:Array<any>,arr2:Array<any>)=>{ 
  if (arr && !arr2||!arr&&arr2)return false
  if (arr && arr2){ 
    if (arr.length!=arr2.length)return false
    for(let i=0;i<=arr.length;i++){
      let a=arr[i]
      let aEq=false;
      for(let j=0;j<=arr2.length;j++){
        console.info(a,"==",arr2[j],a==arr2[j])
        if (a==arr2[j]) {aEq=true;break;}
      }
      if (!aEq)return false
    }
    return true;
  }
  return false
}

const onCheckGroupChange=(result:Array<Number>)=>{ 
  tags.value?.map(m=>m.checked=false) 
  tags.value?.filter(m=>result.indexOf(m.id)>-1) .map(m=>m.checked=true) 
  console.info(array_eq(labelModel.value.dbIds,result),labelModel.value.dbIds,result)
  saveLabelButState.value=!array_eq(labelModel.value.dbIds,result)
}
const saveLabelButState=ref(false) 
//保存标记
const onSaveLabel=()=>{
  if (props.options&& props.options.id){
    let appData=labelModel.value.appendIds.filter(m=>labelModel.value.dbIds.indexOf(m)==-1)
    let removeData=labelModel.value.dbIds.filter(m=>labelModel.value.appendIds.indexOf(m)==-1)
    mark_label_svc(props.options.id,{appendIds:appData,removeIds:removeData}).then(res=>{
      if (res.code==0){
        ElMessage.success(res.message);
        if (props.options&& props.options.id)loadLabel(props.options.id)
      }
      else ElMessage.error(res.message)
    })
  } 
}
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

.ellipsis {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.el-descriptions__body{width: 30%;}
</style>
