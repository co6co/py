
<!--
官网文档:https://element-plus.org/zh-CN/component/tree-v2

-->
<template>
    <div>123456789</div>
    <el-tree-v2
        ref="treeInstance"
        :data="treeData"
        :height="400"
        :indent="26"
        :props="propsMap"
        :show-checkbox="true"
        :default-expand-all="true"
        :highlight-current="true" 
    >
    <template #default="{ node, data }">
        <!-- 目录 -->
        <span v-if="data.children != null" @click="onNodeClick(node, data)" style="display: inline-flex; align-items: center;">
          <i style="display: inline-flex; align-items: center;">
            <svg style="margin: 2px 7px 2px 5px" viewBox="0 0 16 16" width="16" height="16">
              <path d="M14,6 L14,5 L7.58578644,5 L5.58578644,3 L2,3 L2,6 L14,6 Z M14,7 L2,7 L2,13 L14,13 L14,7 Z M1,2 L6,2 L8,4 L15,4 L15,14 L1,14 L1,2 Z" stroke-width="1" fill="#8a8e99" />
            </svg>
          </i> 
          <small :title="node.label">{{ node.label }}</small>
        </span>
 
        <!-- 文档 -->
        <span v-else style="display: inline-flex; align-items: center;">
          <i style="display: inline-flex; align-items: center;">
            <svg style="margin: 2px 5px 2px 3px" viewBox="0 0 16 16" width="16" height="16">
              <path d="M13,6 L9,6 L9,5 L9,2 L3,2 L3,14 L13,14 L13,6 Z M12.5857864,5 L10,2.41421356 L10,5 L12.5857864,5 Z M2,1 L10,1 L14,5 L14,15 L2,15 L2,1 Z" stroke-width="1" fill="#8a8e99" />
            </svg>
          </i> 
          <small :title="node.label">{{ node.label }}</small>
        </span>
      </template> 
    </el-tree-v2>
</template>

<script setup lang="ts">
import {onMounted,onUnmounted,ref,getCurrentInstance,ComponentInternalInstance} from 'vue'
import {ElTreeV2,ElMessage} from 'element-plus';
import { position_svc } from '../../../api/process';


interface Tree{
    id: number,
    name: String,
    children:Tree[]
}
//属性映射
const propsMap={
    children:'children',
    label:'name',
    value:'id',
    disabled:'disabled'
}
const query=(pid:number)=>{
    position_svc({pid:pid}).then( res=>{
    if (res.code==0){
        if (pid==0){
          treeData.value=res.data
        }else{

        }
    }
  })
}
query(0)
//代理对象  as 类型断言
const proxy = getCurrentInstance() as ComponentInternalInstance
//树形实例
const treeInstance =ref(null)
//树数据
const treeData =ref([{id:1,name:'',children:[]}])


const onNodeClick=(node:any,data:Tree)=>{console.info(node,data)}
//选中节点
const onCheckNodesClick=()=>{
    const refs=proxy.refs
    console.log("$refs->",refs) 
    const treeIns=refs.treeInstance as typeof ElTreeV2
    console.log("refs.treeRef->",treeIns)
    const checkedNodes=treeIns.getCheckedNodes()
    console.log("checkedNodes->",checkedNodes) 
}
</script>

