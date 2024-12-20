<template>
  <div>
    <div class="container">
      <div class="handle-box">
        <el-button type="primary">按钮</el-button>
      </div>

      <basis />
      <ss />
      <demo title="标题" content="内容" />
    </div>
  </div>
</template>

<script setup lang="ts" name="export">
import basis from '../components/demo/composition002'
import ss from '@/components/demo/demo004.vue'
import demo from '@/components/demo/demo2'
import { routeHook } from 'co6co-right'
const { getPermissKey } = routeHook.usePermission()
const result = getPermissKey(routeHook.ViewFeature.push)
console.info('11111', result)
</script>
