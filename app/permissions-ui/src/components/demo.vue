<template>
  <div>
    <el-button @click="onClick">{{ btnName }}</el-button>
  </div>
</template>
<script lang="ts" setup>
import { ElMessage } from 'element-plus'

defineProps({
  btnName: {
    type: String,
    default: '按钮'
  }
})
const onClick = () => {
  ElMessage.success('点击了按钮~！')
}
</script>
