import { createRouter, createWebHistory } from 'vue-router'
import { basicRoutes } from './static'
import {type App } from 'vue'
import { setupRouterHooks } from './hooks'
 
export const router = createRouter({ 
  history: createWebHistory(import.meta.env.VITE_UI_PATH), //有二級路徑需要配置，
  routes: basicRoutes
})
 
export default function setupRouter (app: App<Element>): void {
  // 路由钩子函数
  setupRouterHooks()
  app.use(router)
}