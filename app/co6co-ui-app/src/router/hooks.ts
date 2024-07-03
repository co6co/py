import { router, ViewObjects } from './index'

import { usePermissStore, getToken } from 'co6co'
import { routeHook } from 'co6co-right'
import { menuSelectHook } from 'co6co-right'

const { queryRouteData } = routeHook.useRouteData()

// 动态添加路由
export function addRoutes(menu: routeHook.IRouteData[]) {
  menu.forEach((item) => {
    // 只将页面信息添加到路由中
    if (
      item.category == menuSelectHook.MenuCateCategory.VIEW ||
      item.category == menuSelectHook.MenuCateCategory.SubVIEW
    ) {
      const component = ViewObjects[`${item.component}`] //loadView[`../views${e.component}.vue`]
      //console.info("add route",item.name,item.code,"=>",item.component)
      component
        ? router.addRoute('home', {
            name: item.code,
            path: item.url,
            meta: { title: item.name, permiss: item.permissionKey },
            component: component
          })
        : console.warn(`增加路由${item.name}找不到VIEW:${item.component}`)
    }
    if (item.children && item.children.length > 0) addRoutes(item.children)
  })
}

/**
 * 注册新路由
 * 当 用户登录/ 权限改变 路由都可能增加或减少
 * @param bck 回调
 */
export function registerRoute(bck?: () => void) {
  //console.info("to:",to,"from:",from)
  const permiss = usePermissStore()
  //console.info('route..query api...')
  queryRouteData((data: routeHook.IRouteData[], e) => {
    //console.info('route..query api ed...')
    if (data && data.length > 0) {
      const list = routeHook.getAllPermissionKeys(data)
      //console.info("所有权限字",list)
      permiss.set(list)
      addRoutes(data) // 此处的menuList为上述中返回的数据
    } else {
      console.warn('获取路由数据失败或者为空', e)
    }
    if (bck) bck()
  })
}
export function setupRouterHooks() {
  let registerRefesh = true
  //console.info('route..注册或刷新...')
  const permiss = usePermissStore()
  router.beforeEach((to, from, next) => {
    //pageRefresh(to)
    //console.info('route.before...')
    document.title = to.meta.title ? `${to.meta.title}` : import.meta.env.VITE_SYSTEM_NAME
    const token = getToken()
    // console.info( "permiss" ,to.meta.permiss,to.path)

    let pathTemp: string | undefined | any = undefined
    do {
      //未认证 -> login
      if (!token && to.path !== '/login') {
        pathTemp = '/login'
        break
      }
      // 如果没有权限-> 403
      if (to.meta.permiss && !permiss.includes(String(to.meta.permiss))) {
        //removeToken()
        pathTemp = '/403'
        break
      }
      if (registerRefesh) {
        //console.info("to:",to,"from:",from)
        registerRoute(() => {
          registerRefesh = false
          next({ ...to, replace: true })
        })
        return
      }
    } while (false)
    pathTemp ? next(pathTemp) : next()
  })
  router.afterEach((to, form) => {
    localStorage.setItem('new', to.path)
  })
}
