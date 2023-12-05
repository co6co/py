<template>
	<div class="container">  
        <el-collapse v-model="activeNames" @change="handleChange">
            <el-collapse-item title="Consistency" name="1">
                <div>
                Consistent with real life: in line with the process and logic of real
                life, and comply with languages and habits that the users are used to;
                </div>
                <div>
                    Consistent within interface: all elements should be consistent, such
                    as: design style, icons and texts, position of elements, etc.
                </div>
        </el-collapse-item>
        <el-collapse-item  v-for=" (item,index) in data" :key="index" :title="item.name" :name="index">
                <div>
                Consistent with real life: in line with the process and logic of real
                life, and comply with languages and habits that the users are used to;
                </div>
                <div>
                    Consistent within interface: all elements should be consistent, such
                    as: design style, icons and texts, position of elements, etc.
                </div>
        </el-collapse-item>
        </el-collapse> 
	</div>
</template>
<script setup lang="ts">   
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import {ElImage, ElDescriptions,ElDescriptionsItem,ElTag,ElDivider,ElMessage } from 'element-plus';
import {Check,  Delete, Edit,Message,Notebook,Star,Pointer,WarningFilled,UploadFilled,CaretRight} from '@element-plus/icons-vue'
import { fa } from 'element-plus/es/locale';
 
 
const props = defineProps({
  data: {
    type:Array ,
    required: false
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
