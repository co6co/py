import { ref, onMounted } from 'vue'
import * as types from 'co6co'
import { view_route_svc } from '../../api/view'
import { Storage, traverseTreeData, randomString } from 'co6co'

import { useRouter } from 'vue-router'

/**
 * VIEW 功能
 * 为控制页面按钮权限
 * @returns
 */
export enum ViewFeature {
  //** 查看 */
  view = 'view',
  //** 增加 */
  add = 'add',
  //** 编辑 */
  edit = 'edit',
  //** 删除 */
  del = 'del',
  //** 设备 */
  setting = 'setting',
  //** 检查 */
  check = 'check',
  //** 下载 */
  downloads = 'downloads',
  //** 下载 */
  download = 'download',
  //** 关联 */
  associated = 'associated',
  //** 重置 */
  reset = 'reset',
  //** 推送 */
  push = 'push',
  //** 使生效 **/
  effective = 'effective',
  //** 设备编号 */
  settingNo = 'settingNo',
  //** 设置优先级 */
  settingPriority = 'settingPriority'
}
/**
 * 获取当前路由
 * @returns 当前路由
 */
export const getCurrentRoute = () => {
  const router = useRouter()
  // 获取当前路由对象
  const currentRoute = router.currentRoute.value
  return {
    currentRoute
  }
}
/**
 * 获取指定视图权限字
 *
 */
export default getCurrentRoute

const getViewFeature = (pageUrl: string) => {
  const { getRouteData } = useRouteData()
  const data = getRouteData()
  let curremtPageRouteData: Partial<IRouteData> = {}
  if (data)
    traverseTreeData(data, (d) => {
      const item = d as IRouteData
      if (pageUrl && item.url == pageUrl)
        if (item.url == pageUrl) {
          curremtPageRouteData = item
          return true
        }
    })
  return curremtPageRouteData.children
}
/**
 * 获取按钮的权限字
 *
 * @param buttonType  user_add
 * @param pageUrl
 *
 * @returns
 */
const getPermissionKey = (buttonType: ViewFeature | string, pageUrl: string) => {
  let permissionKey: string = ''
  const featureList = getViewFeature(pageUrl)

  if (featureList && featureList.length > 0) {
    const targetKey = `${buttonType}`
    traverseTreeData(featureList, (d) => {
      const item = d as IRouteData
      if (item.methods == targetKey) {
        permissionKey = item.permissionKey
        return true
      }
    })
  }
  if (permissionKey == null) {
    permissionKey = randomString(10)
    console.warn(`${pageUrl}.${buttonType}->permissionKey:${permissionKey}`)
  }
  return permissionKey
}
/**
 * 权限相关
 * @returns
 */
export const usePermission = () => {
  const { currentRoute } = getCurrentRoute()
  const getCurrentViewFeature = () => {
    return getViewFeature(currentRoute.path)
  }
  const getPermissKey = (feature: ViewFeature) => {
    let result = getPermissionKey(feature, currentRoute.path)
    return result
  }

  return { getPermissKey, getCurrentViewFeature }
}

interface RouteItem {
  id: number
  category: number
  parentId: number
  name: string
  code: string
  icon: string
  url: string
  component: string
  methods: string
  permissionKey: string
}

/**
 * 左侧菜单数据类型
 */
export interface sideBarItem {
  icon: string
  index: string
  title: string
  permiss: string
  subs?: Array<sideBarItem>
}

export type IRouteData = Omit<types.ITreeSelect, 'children'> &
  RouteItem & { children?: IRouteData[] }

/**
 * 获取配置的所有权限字
 */
export const getAllPermissionKeys = (data: IRouteData[] | null) => {
  if (!data) {
    const { getRouteData } = useRouteData()
    data = getRouteData()
  }
  let allKeys: string[] = []
  if (data)
    traverseTreeData(data, (d) => {
      const item = d as IRouteData
      if (item.permissionKey && !allKeys.includes(item.permissionKey))
        allKeys.push(item.permissionKey)
    })
  return allKeys
}

/**
 * 通过根据后台登录用户获取菜单Tree
 */
export const useRouteData = () => {
  const Key = 'useRouteData'
  const storage = new Storage()
  const queryRouteData = (bck: (data: IRouteData[], msg?: string) => void) => {
    view_route_svc()
      .then((res) => {
        if (res.code == 0) storage.set<IRouteData[]>(Key, res.data), bck(res.data)
        else bck([], res.message || '请求出错')
      })
      .catch((e) => {
        bck([], e)
      })
  }

  const getRouteData = () => {
    let result = storage.get<IRouteData[]>(Key)
    return result
  }

  return { queryRouteData, getRouteData }
}
