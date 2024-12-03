import { createApp } from 'vue'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App'
import useRouter from './router'

import { co6coPlugin } from 'co6co'
import { setBaseUrl } from './utils'

import 'element-plus/dist/index.css'
import 'co6co/dist/index.css'
import 'co6co-right/dist/index.css'
//import './assets/css/icon.css' 查看在那用 影响全局

const app = createApp(App)
app.use(co6coPlugin, {})
try {
  setBaseUrl()
  app.use(useRouter())
  // 注册图标
  for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
  }
} catch (e) {
  console.error('main:', e)
}
app.mount('#app')
