import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router';
import { usePermissStore } from '../store/permiss';
//import Home from '../views/home.vue';
import wxHome from '../views2/wxHome.vue';
import { getToken, removeToken, setToken } from '../utils/auth';
import { Storage } from '../store/Storage';

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
					title: '用户名管理',
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
					title: '告警信息',
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
					title: '视频列表',
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
					title: '实时视频',
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
			},
			{
				path: '/index.html',
				name: 'wxHome',
				meta: {
					title: '获取用户信息',
					permiss: '2',
				},
				component: () => import('../views2/wxHome.vue'),
			},
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
import { randomString } from '../utils';
import { getRedirectUrl } from '../components/wx';
import { showNotify } from 'vant';
import { ticket_svc } from '../api/authen'; 

const debug = Boolean(Number(import.meta.env.VITE_IS_DEBUG));

const getUrl = () => {
	let url = document.location.toString();
	let arrUrl = url.split('//');
	let start = arrUrl[1].indexOf('/');
	let relUrl = arrUrl[1].substring(start); //stop省略，截取从start开始到结尾的所有字符
	if (relUrl.indexOf('?') != -1) {
		relUrl = relUrl.split('?')[0];
	}
	return relUrl.replace('#', '**');
};

function getQueryVariable(key: string) {
	try {
		//var query = window.location.search.substring(1);
		var query = window.location.href.substring(
			window.location.href.indexOf('?') + 1
		);
		var vars = query.split('&');
		for (var i = 0; i < vars.length; i++) {
			var pair = vars[i].split('=');
			if (pair[0] == key) {
				return pair[1];
			}
		}
		return null;
	} catch (e) {}
	return null;
}

const redirectUrl = () => {
	showNotify({ type: 'warning', message: `跳转...` });
	const redirect_uri = import.meta.env.VITE_WX_redirect_uri;
	const scope = 1;
	let redirectUrl = '';
	if (debug) {
		redirectUrl = getRedirectUrl(
			redirect_uri,
			scope,
			`${randomString(10)}-${scope}-${getUrl()}-${randomString(10)}`
		);
	} else {
		redirectUrl = getRedirectUrl(
			redirect_uri,
			scope,
			`${randomString(10)}-${scope}-${getUrl()}-${randomString(10)}`
		);
	}
	window.location.href = redirectUrl;
};
const ticket = getQueryVariable('ticket');
router.beforeEach((to, from, next) => {
	document.title = `${to.meta.title} `;
	const permiss = usePermissStore();
	let token = getToken();
    console.info(token)
	if (!token) {
		//next('/login');
		if (ticket) {
			ticket_svc(ticket).then((res) => {
				if (res.code == 0) setToken(res.data.token, res.data.expireSeconds);
				else showNotify({ type: 'danger', message: res.message });
			});
		} else {
			redirectUrl();
		}
	}  else {
		next();
	}
});
export default router;
