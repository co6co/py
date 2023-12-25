import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router';
import { usePermissStore } from '../store/permiss'; 
import wxHome from '../views2/wxHome.vue';
import { getToken, removeToken, setToken } from '../utils/auth'; 
import { nextTick } from 'vue'; 

const routes: RouteRecordRaw[] = [
	{
		path: '/',
		redirect: '/usermgr',
	},
	{
		path: '/',
		name: 'Home',
		component: wxHome,
		children: [
			{
				path: '/usermgr',
				name: 'usermgr',
				meta: {
					title: '用户管理',
					permiss: '2',
				},
				component: () => import('../views2/userTable.vue'),
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
				path: '/alarminfo.html',
				name: 'alarmTable',
				meta: {
					title: '告警事件',
					permiss: '2',
				},
				component: () => import('../views2/alarmTable.vue'),
			},
			{
				path: '/alarmdetail.html',
				name: 'alarmdetail',
				meta: {
					title: '告警详情',
					permiss: '2',
				},
				component: () => import('../views2/alarmDetail.vue'),
			},
			{
				path: '/devicelist.html',
				name: 'devicelist',
				meta: {
					title: '设备管理',
					permiss: '2',
				},
				component: () => import('../views2/deviceList.vue'),
			},
			{
				path: '/alarmpreview.html',
				name: 'alarmPreview',
				meta: {
					title: '告警预览',
					permiss: '2',
				},
				component: () => import('../views2/alarmPreview.vue'),
			},

			{
				path: '/preview.html',
				name: 'preview',
				meta: {
					title: '视频监控',
					permiss: '2',
				},
				component: () => import('../views2/devicesPreview.vue'),
			},

			{
				path: '/devicesManage.html',
				name: 'devicesTable',
				meta: {
					title: '设备管理',
					permiss: '2',
				},
				component: () => import('../views2/devicesTable.vue'),
			} 
		],
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
 
import { getQueryVariable } from '../utils';
import { redirectUrl } from '../components/wx';
import { ticket_svc } from '../api/authen';
import { showNotify } from 'vant';
 
const debug = Boolean(Number(import.meta.env.VITE_IS_DEBUG));
const ticket = getQueryVariable('ticket');
router.beforeEach((to, from, next) => {
	document.title = `${to.meta.title} `;
	const permiss = usePermissStore();
	let token = getToken();
	({ type: 'warning', message: token });
	if (!token) { 
		if (ticket) {
			ticket_svc(ticket).then((res) => {
				if (res.code == 0) {
					setToken(res.data.token, res.data.expireSeconds);
					nextTick(()=>next())
					
				}
				else showNotify({ type: 'danger', message: res.message });
			});
		} else {
			redirectUrl();
		} 
		//next('/403');
	} else {
		next();
	}
});
export default router;
