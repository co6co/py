import { createRouter, createWebHistory, Router } from 'vue-router'
import { basicRoutes } from './static'
import { setupRouterHooks } from './hooks'
import { getStoreInstance, ViewComponent } from 'co6co'
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
export { gRouter2 as router }
// vue3 + vite中的动态引入组件的方法
let viewObjects: Record<string, ViewComponent> = import.meta.glob([
  '../views/**/*.vue',
  '../views/**/*.tsx'
])

import { views, moduleName } from 'co6co-right'
import { views as taskViews, moduleName as taskName } from 'co6co-task'
/**
Object.keys(viewObjects).forEach((key) => {
  const viewObject = viewObjects[key]
  console.info('key=>', key, 'viewObject', viewObject, '=>', typeof viewObject)

  viewObject()
    .then((res) => {
      console.info(res)
    })
    .catch((e) => {
      console.info(`解析view:'${key}'\terror,${e}`)
    })
}) */
//const allView = {}
//allView[moduleName] = { ...views }
//allView[taskName] = { ...taskViews }
//Object.keys(allView).forEach((name) => {
//  const moduelView = allView[name]
//  Object.keys(moduelView).forEach((key) => {
//    viewObjects[getViewPath(key, name)] = moduelView[key]
//  })
//})

const store = getStoreInstance()
store.setViews(viewObjects)
store.appendViews(moduleName, views)
store.appendViews(taskName, taskViews)

//console.info(store.views)
export const ViewObjects = store.views
