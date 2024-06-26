<template>
  <!--允许你在应用程序中统一配置Element Plus的一些全局属性和样式，这些配置将被应用于所有Element Plus组件-->
  <el-config-provider :locale="zhCn">
    <router-view />
  </el-config-provider>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElConfigProvider } from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import enUs from 'element-plus/es/locale/lang/en'
const data = ref<{ theme: string; locale: any; size: '' | 'default' | 'small' | 'large' }>({
  theme: 'dark',
  size: '',
  locale: zhCn
})
const changeLang = (locale: 'en' | 'zh') => {
  if (locale === 'en') {
    data.value.locale = enUs
  } else if (locale === 'zh') {
    data.value.locale = zhCn
  }
}
</script>
<style>
@import './assets/css/main.css';
@import './assets/css/color-dark.css';
</style>
