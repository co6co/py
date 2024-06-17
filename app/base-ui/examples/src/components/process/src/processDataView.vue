<template>
	<div>  
        <el-row>
          <el-col :span="24"> 
            <img-video :image-option="imageOption"   :video-options="playerOptions" ></img-video>
          </el-col> 
          <el-col  :span="24">   
            <el-divider  style="margin: 12px 0;" /> 
           
            <el-button v-permiss="getPermissKey(ViewFeature.check)"  type="primary" style="max-width: 0 0 10px 0;" @click="isReaudit=!isReaudit"   > {{isReaudit?"返回":"重审"}}</el-button> 
            <el-button v-permiss="getPermissKey(ViewFeature.download)"  type="primary" :icon="Download" :loading="downloading" style="max-width: 0 0 10px 0;" @click="onDownload"> 下载</el-button> 
            
            <el-descriptions :column="2">
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
            <el-descriptions>  
                <el-descriptions-item label="审核标注"> 
                    {{options?.manualAuditRemark}}
                </el-descriptions-item>
            </el-descriptions> 
          </el-col>
          <el-col :span="24" > 
            <el-button-group  >  
              <el-button @click="onAuditSumit(0)"   type="danger" title="使用 ‘1’ 快捷键"  :disabled="!allowAudit" :icon="WarningFilled"  >误警</el-button><!--0-->
              <el-button @click="onAuditSumit(1)"   type="primary" title="使用 ‘2’ 快捷键" :disabled="!allowAudit" :icon="Check"  >确警</el-button><!--1-->
              <el-button @click="onAuditSumit(2)"  type="success" title="使用 ‘3’ 快捷键" :disabled="!allowAudit" :icon="UploadFilled"  >通过[下发] </el-button><!--2-->
              <el-button @click="onAuditSumit(3)"   type="warning" title="使用 ‘4’ 快捷键"  :disabled="!allowAudit" :icon="Notebook"  >不处理 </el-button> <!--3-->
          </el-button-group>
          </el-col>
        </el-row>  
	</div>
</template>
<script setup lang="ts">   
import { watch, type PropType, ref , computed ,onMounted} from 'vue';  
import { ElDescriptions,ElDescriptionsItem,ElTag,ElDivider,ElMessage } from 'element-plus';

import {  Check,  Delete,  Edit,  Message,   Notebook,  Star,Pointer,WarningFilled,UploadFilled,CaretRight,Download} from '@element-plus/icons-vue'
 
import { imgVideo ,types} from "../../imgvideo";   
import { audit_svc, one_svc,download_one_svc   } from '../../../api/process';
import { request_resource_svc  } from '../../../api'; 
import * as r  from   '../../../api/resource';  

import {type Item} from '../'
import {usePermission,ViewFeature} from '../../../hook/sys/useRoute'
const {getPermissKey} = usePermission()

const props = defineProps({
  options: {
    type:  Object as PropType<Item>,
    required: false
  },
  metaData:{ 
    type:Object as PropType<ItemAattachData>,
    required: true
  },
})   

const emit = defineEmits([ "refesh"])  
//重新审核
const isReaudit=ref(false) // false 审核按钮不能用
//是否运行重新审核  不允许时不显示按钮
let allowReauditBtnStatus=ref(false) 
let allowAudit= computed(()=>{
    if(props.options){   
      return props.options.flowStatus in props.metaData.allowAuditStatus && props.options.manualAuditResult==null || (props.options.flowStatus in props.metaData.allowAuditStatus && isReaudit.value);
    }
    return false;
})  
/** 图片视频显示 */ 
/**   图片视频显示 */ 
const imageOption=ref<Array<types.imageOption> >([{url:"",thumbnail:""} ,{url:"",thumbnail:""} ] )
const playerOptions=ref<Array<types.videoOption> >([{url:"",poster:""},{url:"",poster:""}] )
const updatePlayerOptions=(index:number,address?:string)=>{
  if (address){ 
        playerOptions.value[index].url=r.get_video_url(address)
        playerOptions.value[index].poster=r.get_poster_url(address) 
    }else {playerOptions.value[index].poster="" , playerOptions.value[index].url=""}
}
const updateImageOptions=(index:number,address?:string)=>{
  if (address){ 
    imageOption.value[index].url=r.get_img_url(address)
    imageOption.value[index].thumbnail=r.get_thumbnail_url(address) 
    }else {imageOption.value[index].thumbnail="" , imageOption.value[index].url=""}
}
watch(()=>props.options?.videoSavePath, (n?:string,o?:string)=>{updatePlayerOptions (0,n) })
watch(()=>props.options?.annoVideoSavePath, (n?:string,o?:string)=>{  updatePlayerOptions(1,n) }) 

watch(()=>props.options?.pic1SavePath, (n?:string,o?:string)=>{ updateImageOptions(0,n) })
watch(()=>props.options?.annoPic1SavePath, (n?:string,o?:string)=>{  updateImageOptions(1,n) }) 
/** end 图片视频显示 */
const keyDown=(e:KeyboardEvent)=>{
  const keys=["1","2","3","4"] 
  if (  keys.indexOf( e.key) >-1){     
    onAuditSumit(keys.indexOf(e.key)); 
  } 
  e.stopPropagation()
}
onMounted(()=>{ 
  //会触发其他未知现象
  //document.addEventListener("keyup", keyDown)
  //onBeforeUnmount(()=>{ document.removeEventListener("keyup", keyDown)})  
})


const onAuditSumit=(statue:number)=>{
  if(allowAudit.value){
    isReaudit.value=false
    let id=props.options?.id;  
    audit_svc({"id":id,"statue":statue}).then(res=>{
      if(res.code==0){
        ElMessage.success(res.message), 
        emit('refesh')
      }
      else {ElMessage.error(res.message)}
    })  
  }else{
    ElMessage.error("当前状态不允许审核！")
  } 
} 

let info=ref<{auditUser:{isCurrentUser:boolean,isAdminstrator:boolean}}> ({auditUser:{isCurrentUser:false,isAdminstrator:false}}) 
watch(()=> [props.options?.id,props.options?.manualAuditResult] , async ([neId,nestatue],[oldId,oldStatue]) => { 
  if(neId) {
    info.value =(await one_svc(neId)).data; 
    allowReauditBtnStatus.value=nestatue!= null&&(info.value.auditUser.isCurrentUser || info.value.auditUser.isAdminstrator)
  }
})
const downloading=ref(false)
const onDownload=()=>{
  if(props.options){
    downloading.value=true;
    download_one_svc(props.options.id,props.options,()=>{ downloading.value=false})
  }
  else ElMessage.error("数据未加载不能下载")
}
defineExpose({ keyDown })
</script>

<style scoped lang="less"> 
.ellipsis {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.el-descriptions__body{width: 30%;}
</style>
