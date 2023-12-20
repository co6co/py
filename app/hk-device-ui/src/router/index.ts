import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router';
import { usePermissStore } from '../store/permiss';
import Home from '../views/home.vue';

import { getToken, removeToken } from '../utils/auth';
import { Storage } from '../store/Storage';

const storeage=new Storage()
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
                path: '/taskTable',
                name: 'taskTable',
                meta: {
                    title: '任务管理',
                    permiss: '2',
                },
                component: () => import( '../views/taskTable.vue'),
            }, 
            {
                path: '/devicesTable',
                name: 'devicesTable',
                meta: {
                    title: '设备管理',
                    permiss: '2',
                },
                component: () => import( '../views/devicesTable.vue'),
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
        component: () => import( '../views/403.vue'),
    },
];

const router = createRouter({
    history: createWebHashHistory(),
    routes,
});

router.beforeEach((to, from, next) => {
    document.title = `${to.meta.title} `; 
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
