import { type RouteRecordRaw } from 'vue-router'

const Login = () => import('@/views/login.vue')
const page403 = () => import('@/views/403.vue')
const home =()=> import('@/views/home.vue')

// 基础路由，不需要设置权限
export const basicRoutes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/processAudit', //processAudit
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
    path: '/home',
    name: 'home',
    component: home
  }
]
