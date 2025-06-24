import { basicRoutes } from './static'
import { getStoreInstance, ViewComponent } from 'co6co'
import { routeHook } from 'co6co-right'

// vue3 + vite中的动态引入组件的方法
let viewObjects: Record<string, ViewComponent> = import.meta.glob([
  '../views/**/*.vue',
  '../views/**/*.tsx'
])

import { views, moduleName } from 'co6co-right'
import { views as taskViews, moduleName as taskName } from 'co6co-task'
const store = getStoreInstance()
store.setViews(viewObjects)

store.appendViews(moduleName, views)
store.appendViews(taskName, taskViews)

//console.info(store.views)
//export const ViewObjects = store.views
const base = `${import.meta.env.VITE_UI_PATH}${import.meta.env.VITE_UI_PC_INDEX}`

export default routeHook.CreateRouter(base, basicRoutes)
