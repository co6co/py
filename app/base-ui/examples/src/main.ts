import { createApp } from 'vue'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App.vue'
import setupRouter from './router'
import { makeInstaller } from 'co6co'
import 'element-plus/dist/index.css'
import './assets/css/icon.css'

const app = createApp(App)
try {
  const { install, version } = makeInstaller()
  install(app)
  //app.use(router);
  setupRouter(app)
  console.info('version：', version)
  // 注册图标
  for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
  }
} catch (e) {
  console.error('main:', e)
}

// 自定义权限指令
/*
const permiss = usePermissStore()
app.directive('permiss', {
  mounted(el, binding: any) {
    if (!permiss.includes(String(binding.value))) {
      el['hidden'] = true
    }
  }
})*/
app.mount('#app')
