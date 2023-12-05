import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router';
import { usePermissStore } from '../store/permiss';
import Home from '../views/home.vue';

const routes: RouteRecordRaw[] = [
    {
        path: '/',
        redirect: '/usermgr',
    },
    {
        path: '/',
        name: 'Home',
        component: Home,
        children: [
            /*
            {
                path: '/dashboard',
                name: 'dashboard',
                meta: {
                    title: '系统首页',
                    permiss: '1',
                },
                component: () => import(  '../views/dashboard.vue'),
            },
            {
                path: '/table',
                name: 'basetable',
                meta: {
                    title: '表格',
                    permiss: '2',
                },
                component: () => import(  '../views/table.vue'),
            },
            {
                path: '/charts',
                name: 'basecharts',
                meta: {
                    title: '图表',
                    permiss: '11',
                },
                component: () => import(  '../views/charts.vue'),
            },
            {
                path: '/form',
                name: 'baseform',
                meta: {
                    title: '表单',
                    permiss: '5',
                },
                component: () => import( '../views/form.vue'),
            },
            {
                path: '/tabs',
                name: 'tabs',
                meta: {
                    title: 'tab标签',
                    permiss: '3',
                },
                component: () => import(  '../views/tabs.vue'),
            },
            {
                path: '/donate',
                name: 'donate',
                meta: {
                    title: '鼓励作者',
                    permiss: '14',
                },
                component: () => import(  '../views/donate.vue'),
            },
            {
                path: '/permission',
                name: 'permission',
                meta: {
                    title: '权限管理',
                    permiss: '13',
                },
                component: () => import(  '../views/permission.vue'),
            },
            {
                path: '/upload',
                name: 'upload',
                meta: {
                    title: '上传插件',
                    permiss: '6',
                },
                component: () => import(  '../views/upload.vue'),
            },
            {
                path: '/icon',
                name: 'icon',
                meta: {
                    title: '自定义图标',
                    permiss: '10',
                },
                component: () => import( '../views/icon.vue'),
            },
            {
                path: '/user',
                name: 'user',
                meta: {
                    title: '个人中心',
                },
                component: () => import(  '../views/user.vue'),
            },
            {
                path: '/editor',
                name: 'editor',
                meta: {
                    title: '富文本编辑器',
                    permiss: '8',
                },
                component: () => import(  '../views/editor.vue'),
            },
            {
                path: '/markdown',
                name: 'markdown',
                meta: {
                    title: 'markdown编辑器',
                    permiss: '9',
                },
                component: () => import(  '../views/markdown.vue'),
            },
            {
                path: '/export',
                name: 'export',
                meta: {
                    title: '导出Excel',
                    permiss: '2',
                },
                component: () => import(  '../views/export.vue'),
            },
            {
                path: '/import',
                name: 'import',
                meta: {
                    title: '导入Excel',
                    permiss: '2',
                },
                component: () => import(  '../views/import.vue'),
            },
            */
            {
                path: '/usermgr',
                name: 'usermgr',
                meta: {
                    title: '用户名管理',
                    permiss: '2',
                },
                component: () => import(  '../views/userTable.vue'),
            },
            {
                path: '/user',
                name: 'user',
                meta: {
                    title: '个人中心',
                },
                component: () => import(  '../views/user.vue'),
            },
            {
                path: '/processAudit',
                name: 'processAudit',
                meta: {
                    title: '数据审核',
                    permiss: '2',
                },
                component: () => import(/* webpackChunkName: "process" */ '../views/processTable.vue'),
            },
            {
                path: '/taskTable',
                name: 'taskTable',
                meta: {
                    title: '任务列表',
                    permiss: '2',
                },
                component: () => import(/* webpackChunkName: "process" */ '../views/taskTable.vue'),
            },
            {
                path: '/groupTable',
                name: 'groupTable',
                meta: {
                    title: '分组信息',
                    permiss: '2',
                },
                component: () => import(/* webpackChunkName: "process" */ '../views/groupTable.vue'),
            },
            {
                path: '/wxMenuTable',
                name: 'wxMenuTable',
                meta: {
                    title: '公告号菜单管理',
                    permiss: '2',
                },
                component: () => import(/* webpackChunkName: "process" */ '../views/wxMenuTable.vue'),
            },
            {
                path: '/alarmTable',
                name: 'alarmTable',
                meta: {
                    title: '数据管理',
                    permiss: '2',
                },
                component: () => import(/* webpackChunkName: "process" */ '../views/alarmTable.vue'),
            }, 
            {
                path: '/devicesTable',
                name: 'devicesTable',
                meta: {
                    title: '设备管理',
                    permiss: '2',
                },
                component: () => import(/* webpackChunkName: "process" */ '../views/devicesTable.vue'),
            }, 
        ],
    },
    {
        path: '/login',
        name: 'Login',
        meta: {
            title: '登录',
        },
        component: () => import(/* webpackChunkName: "login" */ '../views/login.vue'),
    },
    {
        path: '/403',
        name: '403',
        meta: {
            title: '没有权限',
        },
        component: () => import(/* webpackChunkName: "403" */ '../views/403.vue'),
    },
];

const router = createRouter({
    history: createWebHashHistory(),
    routes,
});

router.beforeEach((to, from, next) => {
    document.title = `${to.meta.title} `;
    const role = localStorage.getItem('ms_username');
    const permiss = usePermissStore();
    if (!role && to.path !== '/login') {
        next('/login');
    } else if (to.meta.permiss && !permiss.key.includes(to.meta.permiss)) {
        // 如果没有权限，则进入403
        next('/403');
    } else {
        next();
    }
});

export default router;
