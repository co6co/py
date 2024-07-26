import { createApp } from 'vue'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App.vue'
import useRouter from './router'

import { installPermissDirective, piniaInstance } from 'co6co'
import { setBaseUrl } from './utils'

import 'md-editor-v3/lib/style.css'
import 'co6co/dist/index.css'
import 'co6co-right/dist/index.css'
import 'co6co-wx/dist/index.css'
import 'element-plus/dist/index.css'
import './assets/css/icon.css'

const app = createApp(App)
app.use(piniaInstance)
app.use(installPermissDirective, { instance: piniaInstance })
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
