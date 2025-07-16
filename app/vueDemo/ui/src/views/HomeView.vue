<template>
  <div class="bg-white rounded-lg shadow-lg p-6">
    <h2 class="text-xl font-semibold text-gray-800 mb-4">欢迎来到首页</h2>
    <p class="text-gray-600">这是一个基于 Vue 3 和 Tailwind CSS 的项目模板。</p>
    
    <div class="mt-6">
      <router-link to="/about" class="inline-flex items-center px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors">
        <i class="fa fa-info-circle mr-2"></i> 关于页面
      </router-link>
    </div>
  </div>
</template>

<script setup>
// 首页逻辑
</script>
  