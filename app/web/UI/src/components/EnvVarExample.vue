<template>
  <div class="env-var-example">
    <h3>全局变量使用示例</h3>
    
    <!-- 方式1：直接在模板中使用 -->
    <div class="example-item">
      <strong>方式1 - 直接在模板中使用：</strong>
      <p>API_URL: {{ window._env_.API_URL }}</p>
    </div>
    
    <!-- 方式2：通过computed属性使用 -->
    <div class="example-item">
      <strong>方式2 - 通过computed属性使用：</strong>
      <p>API_URL: {{ apiUrl }}</p>
    </div>
    
    <!-- 方式3：通过data属性使用 -->
    <div class="example-item">
      <strong>方式3 - 通过data属性使用：</strong>
      <p>API_URL: {{ apiUrlFromData }}</p>
    </div>
    
    <!-- 方式4：在setup函数中使用 -->
    <div class="example-item">
      <strong>方式4 - 在setup函数中使用：</strong>
      <p>API_URL: {{ apiUrlFromSetup }}</p>
    </div>
    
    <!-- 展示所有环境变量 -->
    <div class="example-item">
      <strong>所有环境变量：</strong>
      <pre>{{ JSON.stringify(window._env_, null, 2) }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

// 方式4：在setup函数中直接访问全局变量
const apiUrlFromSetup = ref(window._env_.API_URL)

// 方式2：通过computed属性访问
const apiUrl = computed(() => {
  return window._env_.API_URL
})

// 方式3：在data属性中使用（注意：在setup中使用ref替代data）
const apiUrlFromData = ref(window._env_.API_URL)

// 在生命周期钩子中使用
onMounted(() => {
  console.log('Environment variables:', window._env_)
  console.log('API_URL from onMounted:', window._env_.API_URL)
})
</script>

<script lang="ts">
export default {
  name: 'EnvVarExample',
  // 传统Options API方式
  data() {
    return {
      // 方式3：在data中访问全局变量
      apiUrlFromOptionsData: window._env_.API_URL
    }
  },
  computed: {
    // 方式2：在computed中访问全局变量
    apiUrlFromOptionsComputed() {
      return window._env_.API_URL
    }
  },
  mounted() {
    // 在生命周期钩子中访问
    console.log('API_URL from Options API mounted:', window._env_.API_URL)
  }
}
</script>

<style scoped>
.env-var-example {
  padding: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  margin: 20px;
  background-color: #f9f9f9;
}

.example-item {
  margin-bottom: 15px;
  padding: 10px;
  background-color: white