import { createApp } from 'vue';
// 1. 引入你需要的组件
import { Button  } from 'vant'; 
// 2. 引入组件样式
import 'vant/lib/index.css';
import App from './mpApp.vue';
const app = createApp(App);

// 3. 注册你需要的组件
app.use(Button);

import { createPinia } from 'pinia';
import router from './router/home';
import { usePermissStore } from './store/permiss';
 
import './assets/css/icon.css'; 


app.use(createPinia());
app.use(router); 

import * as ElementPlusIconsVue from '@element-plus/icons-vue';
// 注册elementplus图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component);
}
 
// 自定义权限指令
const permiss = usePermissStore();
app.directive('permiss', {
    mounted(el, binding) {
        if (!permiss.key.includes(String(binding.value))) {
            el['hidden'] = true;
        }
    },
}); 
app.mount('#app');