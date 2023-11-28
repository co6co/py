<template>  
    <div class="tree_box" :style="width && {width: width.includes('px') ? width : width + 'px'}">
      <el-select v-model="select.value" clearable filterable ref="treeSelect" :placeholder="placeholder || '请选择'" 
      :filter-method="selectFilter" @clear="clearSelected">
        <el-option :value="select.currentNodeKey" :label="select.currentNodeLabel">
          <el-tree-v2 id="tree_v2"  :ref="treeInstance"
           :data="options"
           :props="keyProps || TreeProps"
           :height="240"
           :width="350"
           :current-node-key="select.currentNodeKey"
           @node-click="onNodeClick"
           @check="onNodeCheck"
           :expand-on-click-node="false"
           :show-checkbox="true"
           :filter-method="treeFilter"
           lazy
           :load="onLoad"
          ></el-tree-v2>
        </el-option>
      </el-select>
      <el-tree-select
        v-model="value"
        :show-checkbox="true"
        lazy
        :load="load"
        :props="props2"
        :cache-data="cacheData"
      />
    </div>
  </template>

<script lang="ts" setup  >
import { TreeNode } from 'element-plus/es/components/tree-v2/src/types'
import { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import {  ref,  defineComponent, nextTick, onMounted,  PropType, reactive,toRefs, watch } from 'vue'
import { PropsIter ,TreeItem } from './types'
const value = ref()
const props2 = {
  label: 'label',
  children: 'children',
  isLeaf: 'isLeaf',
}

let id = 0
const cacheData = [{ value: 5, label: '位置信息' }]
const load =(node:TreeNodeData, resolve:any) => {
  if (node.isLeaf) return resolve([]) 
  setTimeout(() => {
    resolve([
      {
        value: ++id,
        label: `lazy load node${id}`,
      },
      {
        value: ++id,
        label: `lazy load node${id}`,
        isLeaf: true,
      },
    ])
  }, 400)
}
 
const TreeProps:PropsIter = {
  value: 'id',
  label: 'label',
  children: 'children',
  isLeaf:"isLeaf"
} 
const props =defineProps({
    // 组件绑定的options
    options: {
      type: Array as PropType<TreeItem[]>,
      required: true
    },
    // 配置选项
    keyProps: Object as PropType<PropsIter>,
    // 双向绑定值
    modelValue: [String, Number],
    // 组件样式宽
    width: String,
    // 空占位字符
    placeholder: String
})
const { modelValue } = toRefs(props) 
//树形实例
const treeInstance =ref(null)


const select:any = reactive({
    value:props.modelValue,
    currentNodeKey: '',
    currentNodeLabel: ''
})
const treeSelect = ref<HTMLElement | null>(null)
// eslint-disable-next-line @typescript-eslint/no-unused-vars


const emit = defineEmits(['update:modelValue',"nodeCheck","loadData"]) 
const onNodeClick = (data: TreeNodeData, node: TreeNode) => {
    console.info(data,node)
    select.currentNodeKey = data.id
    select.currentNodeLabel = data.label
    select.value = data.id;
    (treeSelect.value as any).blur()
    emit('update:modelValue', select.value)  
}
const onLoad= (data: TreeNodeData, resolve: any) => {emit('loadData', data,resolve) }
const onNodeCheck= (data: TreeNodeData, node: TreeNode ) => { 
    /*
    select.currentNodeKey = data.id
    select.currentNodeLabel = data.label
    select.value = data.id;
    (treeSelect.value as any).blur()
    emit('update:modelValue', select.value)
    */
    emit('nodeCheck', data,node,treeInstance)
}

 
// select 筛选方法 treeV2 refs
const treeV2:any = ref<HTMLElement | null>(null)
const selectFilter = (query:string) => {
    treeV2.value.filter(query)
}
// ztree-v2 筛选方法
const treeFilter = (query:string, node: TreeNode) => {
    return node.label?.indexOf(query) !== -1
}
// 直接清空选择数据
const clearSelected = () => {
    select.currentNodeKey = ''
    select.currentNodeLabel = ''
    select.value = ''
    emit('update:modelValue', null)
}
// setCurrent通过select.value 设置下拉选择tree 显示绑定的v-model值
// 可能存在问题：当动态v-model赋值时 options的数据还没有加载完成就会失效，下拉选择时会警告 placeholder
const setCurrent = () => {
    select.currentNodeKey = select.value
    treeV2.value.setCurrentKey(select.value)
    const data:TreeNodeData | undefined = treeV2.value.getCurrentNode(select.value)
    select.currentNodeLabel = data?.label
}
// 监听外部清空数据源 清空组件数据
watch([() => props.modelValue], (v) => {
    if (v === undefined && select.currentNodeKey !== '') {
        clearSelected()
    }
    // 动态赋值
    if (v) {
        select.value = v
        setCurrent()
    }
})
// 回显数据
onMounted(async () => {
    await nextTick()
    if (select.value)   setCurrent()
}) 
</script> 

<style lang="less" scoped>
.tree_box{
  width: 214px;
}
.el-scrollbar .el-scrollbar__view .el-select-dropdown__item {
  height: auto;
  max-height: 274px;
  padding: 0;
  overflow: hidden;
  overflow-y: auto;
}

.el-select-dropdown__item.selected {
  font-weight: normal;
}

ul li :deep(.el-tree .el-tree-node__content) {
  height: auto;
  padding: 0 20px;
}

.el-tree-node__label {
  font-weight: normal;
}

.el-tree :deep(.is-current .el-tree-node__label) {
  color: #409eff;
  font-weight: 700;
}

.el-tree :deep(.is-current .el-tree-node__children .el-tree-node__label) {
  color: #606266;
  font-weight: normal;
}
.selectInput {
  padding: 0 5px;
  box-sizing: border-box;
}
.el-select{
  width: 100% !important;
}
</style> 