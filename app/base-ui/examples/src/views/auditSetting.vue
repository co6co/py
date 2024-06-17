<template>
  <div class="container-layout">
    <el-container>
      <el-main>
        <el-scrollbar>
          <setting-audit></setting-audit>
        </el-scrollbar>
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
 
import settingAudit from '../components/biz/settingAudit'
</script>

<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
