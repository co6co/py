import { createRouter, createWebHistory, Router } from 'vue-router'
import { basicRoutes } from './static'
import { setupRouterHooks } from './hooks'
import { registerRoute } from 'co6co-right'
import { getStoreInstance } from 'co6co'
let gRouter2: Router

export default function useRouter(): Router {
  const router = createRouter({
    history: createWebHistory(`${import.meta.env.VITE_UI_PATH}${import.meta.env.VITE_UI_PC_INDEX}`), //有二級路徑需要配置，
    routes: basicRoutes
  })
  // 路由钩子函数
  setupRouterHooks(router)
  gRouter2 = router
  return router
}
export { gRouter2 as router, registerRoute }
// vue3 + vite中的动态引入组件的方法
let viewObjects = import.meta.glob(['../views/**/*.vue', '../sysViews/**/*.vue'])
import { views, getViewPath } from 'co6co-right'
import { views as wxViews, getViewPath as wxGetViewPath } from 'co6co-wx'

Object.keys(views).forEach((key) => {
  viewObjects[getViewPath(key)] = views[key]
})
Object.keys(wxViews).forEach((key) => {
  viewObjects[wxGetViewPath(key)] = wxViews[key]
})
const store = getStoreInstance()
store.setViews(viewObjects)
//console.info(viewObjects)
export const ViewObjects = viewObjects
