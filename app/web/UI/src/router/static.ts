import { type RouteRecordRaw } from 'vue-router'

const Login = () => import('../views/login.vue')
const page403 = () => import('../views/403.vue')
const page404 = () => import('../views/404.vue')
const home = () => import('../views/home.vue')
const markdown = () => import('../views/markdown.vue')
const demo = () => import('../views/demo.tsx')
import { UserTableView } from 'co6co-right'

// 基础路由，不需要设置权限
export const basicRoutes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/usermgr' //processAudit
  },
  {
    path: '/login',
    name: 'login',
    meta: { title: '登录' },
    component: Login
  },
  {
    path: '/403',
    name: '403',
    meta: { title: '没有权限' },
    component: page403
  },
  {
    path: '/404',
    name: '404',
    meta: { title: '未找到' },
    component: page404
  },
  {
    path: '/home',
    name: 'home',
    component: home
  },
  {
    path: '/markdown',
    name: 'markdown',
    component: markdown
  },
  {
    path: '/sysDemo',
    name: 'sysDemo',
    component: UserTableView
  },
  {
    path: '/demo',
    name: 'demo',
    component: demo
  }
]
