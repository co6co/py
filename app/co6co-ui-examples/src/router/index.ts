import { createRouter, createWebHistory } from 'vue-router'
import { basicRoutes } from './static'
import { type App } from 'vue'
import { setupRouterHooks } from './hooks'
import { getStoreInstance } from 'co6co'

export const router = createRouter({
  history: createWebHistory(import.meta.env.VITE_UI_PATH), //有二級路徑需要配置，
  routes: basicRoutes
})

export default function setupRouter(app: App<Element>): void {
  // 路由钩子函数
  setupRouterHooks()
  app.use(router)
}

// vue3 + vite中的动态引入组件的方法
let viewObjects = import.meta.glob(['../views/**/*.vue', '../sysViews/**/*.vue'])
import { views, getViewPath } from 'co6co-right'
import { views as vxView, getViewPath as getWxViewPath } from 'co6co-wx'

Object.keys(views).forEach((key) => {
  viewObjects[getViewPath(key)] = views[key]
})

console.info('OBJECT', vxView)
Object.keys(vxView).forEach((key) => {
  viewObjects[getWxViewPath(key)] = vxView[key]
})
const store = getStoreInstance()
store.setViews(viewObjects)
export const ViewObjects = viewObjects
