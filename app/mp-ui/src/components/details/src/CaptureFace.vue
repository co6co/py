<template>
	<div class="container">  
        <el-collapse v-model="activeNames" @change="handleChange"> 
          <el-collapse-item  v-for=" (item,index) in data" :key="index" :title="item.name" :name="index"> 
            <div> 
                <el-descriptions 
                  direction="vertical"
                  :column="3" 
                  border
                >
                <el-descriptions-item   :key="index2" :label="key.toString()" v-for="(value,key,index2) in item.data">{{value }}</el-descriptions-item> 
              </el-descriptions>  
              </div> 
          </el-collapse-item>
        </el-collapse> 
	</div>
</template>
<script setup lang="ts">   
import { watch, type PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import {ElImage, ElDescriptions,ElDescriptionsItem,ElTag,ElDivider,ElMessage } from 'element-plus';
import {Check,  Delete, Edit,Message,Notebook,Star,Pointer,WarningFilled,UploadFilled,CaretRight} from '@element-plus/icons-vue'
 
  
interface detailsInfo{
    name:string
    data:any
}
const props = defineProps({
  data: {
    type:Array<detailsInfo> ,
    required: true
  } 
})
const activeNames = ref(['1'])
const handleChange = (val: string[]) => {
  console.log(val)
}
//const emit = defineEmits([ "refesh"])
  
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
