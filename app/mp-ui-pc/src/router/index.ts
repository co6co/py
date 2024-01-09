import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router';
import { usePermissStore } from '../store/permiss';
import Home from '../views/home.vue';

import { getToken, removeToken } from '../utils/auth';
import { Storage } from '../store/Storage';

const storeage = new Storage();
const routes: RouteRecordRaw[] = [
	{
		path: '/',
		redirect: '/devicesPreview',
	},
	{
		path: '/',
		name: 'Home',
		component: Home,
		children: [
			{
				path: '/usermgr',
				name: 'usermgr',
				meta: {
					title: '用户管理',
					permiss: '2',
				},
				component: () => import('../views/userTable.vue'),
			},
			{
				path: '/user',
				name: 'user',
				meta: {
					title: '个人中心',
				},
				component: () => import('../views/user.vue'),
			},

			{
				path: '/wxMenuTable',
				name: 'wxMenuTable',
				meta: {
					title: '菜单管理',
					permiss: '2',
				},
				component: () => import('../views/wxMenuTable.vue'),
			},
			{
				path: '/alarmTable',
				name: 'alarmTable',
				meta: {
					title: '告警事件',
					permiss: '2',
				},
				component: () => import('../views/alarmIncident.vue'),
			},
			{
				path: '/devicesPreview',
				name: 'devicesPreview',
				meta: {
					title: '视频监控',
					permiss: '2',
				},
				component: () => import('../views/devicesPreview.vue'),
			},
			{
				path: '/siteMgr',
				name: 'siteMgr',
				meta: {
					title: '站点管理',
					permiss: '2',
				},
				component: () => import('../views/siteMgr.vue'),
			},
			{
				path: '/deviceMgr',
				name: 'deviceMgr',
				meta: {
					title: '设备管理',
					permiss: '2',
				},
				component: () => import('../views/deviceMgr.vue'),
			},
		],
	},
	{
		path: '/login',
		name: 'Login',
		meta: {
			title: '登录',
		},
		component: () => import('../views/login.vue'),
	},
	{
		path: '/403',
		name: '403',
		meta: {
			title: '没有权限',
		},
		component: () => import('../views/403.vue'),
	},
]; 
const router = createRouter({
	history: createWebHashHistory(),
	routes,
});

router.beforeEach((to, from, next) => {
	document.title = `${to.meta.title} `;
	const role = storeage.get('username');
	const permiss = usePermissStore();
	let token = getToken();
	if (!token && to.path !== '/login') {
		next('/login');
	} else if (to.meta.permiss && !permiss.key.includes(to.meta.permiss)) {
		// 如果没有权限，则进入403
		next('/403');
	} else {
		next();
	}
});

export default router;
