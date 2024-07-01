import { createApp } from 'vue'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App.vue'
import setupRouter from './router'
//import { usePermiss } from 'co6co'

import { installPermissDirective, piniaInstance } from 'co6co'
import 'co6co-right/dist/index.css'
import 'element-plus/dist/index.css'
import './assets/css/icon.css'

const app = createApp(App)
app.use(piniaInstance)
app.use(installPermissDirective, { instance: piniaInstance })

//app.use(usePermiss)
try {
  //const { install, version } = makeInstaller()
  //install(app)
  //app.use(router);
  app.config.globalProperties.$baseUrl = import.meta.env.VITE_BASE_URL
  setupRouter(app)
  //console.info('version：', version)
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
