import { createRouter, createWebHistory,type RouteRecordRaw} from 'vue-router';
import { usePermissStore } from '../store/permiss'; 
import wxHome from '../views2/wxHome.vue';
import { getToken, removeToken, setToken } from '../utils/auth'; 
import { nextTick } from 'vue'; 
import { Storage,SessionKey } from '../store/Storage';

let storeage = new Storage();
const routes: RouteRecordRaw[] = [
	{
		path: '/',
		redirect: '/devicesPreview.html',
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
				path: '/devicelistOld.html',
				name: 'devicelistOld',
				meta: {
					title: '视频监控',
					permiss: '2',
				},
				component: () => import('../views2/deviceList.vue'),
			},
			{
				path: '/devicesPreview.html',
				name: 'devicesPreview2',
				meta: {
					title: '视频监控',
					permiss: '2',
				},
				component: () => import('../views2/devicesPreview2.vue'),
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
	history:  createWebHistory("/home.html/#/"), 
	routes,
});
console.info(router)
 
import { getQueryVariable } from '../utils';
import { redirectUrl } from '../components/wx';
import { ticket_svc } from '../api/authen';
import { showNotify } from 'vant'; 
 
const debug = Boolean(Number(import.meta.env.VITE_IS_DEBUG));
const appid = import.meta.env.VITE_ENV;
const ticket = getQueryVariable('ticket');
router.beforeEach((to, from, next) => {
	document.title = `${to.meta.title} `;
	const permiss = usePermissStore();
	let token = getToken();
	//console.info({ type: 'warning', token: token },to.fullPath); 
	if (!token) {  
		if (ticket) {
			ticket_svc(ticket).then((res) => {
				if (res.code == 0) {
					setToken(res.data.token, res.data.expireSeconds); 
					storeage.set(SessionKey, res.data.sessionId, res.data.expireSeconds);
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
