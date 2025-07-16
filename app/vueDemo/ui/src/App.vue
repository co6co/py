<template>
  <div class="container mx-auto px-4 py-8"  >
    <header class="mb-8">
      <h1 class="text-3xl font-bold text-gray-800">Vue 3 项目</h1>
      <p class="text-gray-600 mt-2">使用 Vite 和 Tailwind CSS 构建</p>
    </header> 
    <main>
      <nav class="mb-6">
        <ul class="flex space-x-4">
          <li><router-link to="/" class="text-blue-600 hover:underline">首页</router-link></li>
          <li><router-link to="/about" class="text-blue-600 hover:underline">关于</router-link></li>
          <li><router-link to="/chart" class="text-blue-600 hover:underline">图表</router-link></li>
        </ul>
      </nav>
      <router-view />
    </main>
  </div>
</template>

 
<script setup>
 
import { useRouter,RouterView ,RouterLink} from 'vue-router'  
const router = useRouter()  
</script>
 
 

<style scoped>
/* 自定义样式 */
nav ul {
  list-style-type: none;
}

nav ul li {
  transition: all 0.3s ease;
  cursor: pointer;
}

nav ul li a.router-link-exact-active {
  color: #1e40af;
  font-weight: bold;
}

nav ul li a:hover {
  text-decoration: underline;
  color: #1d4ed8;
}

nav ul li a {
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
}

nav ul li a:hover:not(.router-link-exact-active) {
  background-color: #f3f4f6;
} 
</style>
