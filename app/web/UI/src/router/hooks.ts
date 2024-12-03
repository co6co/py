import { Router } from 'vue-router'
import { ViewObjects } from './index'

import { usePermissStore, getToken } from 'co6co'
import { registerRoute, validAuthenticate } from 'co6co-right'
import { ElMessage } from 'element-plus'

function jump(next: (path?: string) => void, path?: string) {
  path ? next(path) : next()
}

export function setupRouterHooks(router: Router) {
  let registerRefesh = true
  const permiss = usePermissStore()
  router.beforeEach((to, _, next) => {
    document.title = to.meta.title ? `${to.meta.title}` : import.meta.env.VITE_SYSTEM_NAME
    if (registerRefesh) {
      registerRoute(ViewObjects, router, (msg) => {
        if (msg && getToken()) {
          ElMessage.error(msg)
        }
        registerRefesh = false
        next({ ...to, replace: true })
      })
    } else {
      let pathTemp: string | undefined | any = undefined
      validAuthenticate(
        () => {
          if (to.meta.permiss && !permiss.includes(String(to.meta.permiss))) {
            pathTemp = '/403'
          }
          jump(next, pathTemp)
        },
        () => {
          if (to.path !== '/login') pathTemp = '/login'
          jump(next, pathTemp)
        }
      )
    }
  })
  router.afterEach((to) => {
    localStorage.setItem('new', to.path)
  })
}
