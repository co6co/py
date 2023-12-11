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
                path: '/alarmTable',
                name: 'alarmTable',
                meta: {
                    title: '数据管理',
                    permiss: '2',
                },
                component: () => import( '../views/alarmTable.vue'),
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
    const role = localStorage.getItem('ms_username');
    const permiss = usePermissStore();
    if (!role && to.path !== '/login') {
        //next('/login');
        next();
    } else if (to.meta.permiss && !permiss.key.includes(to.meta.permiss)) {
        // 如果没有权限，则进入403
        next('/403');
    } else {
        next();
    }
});

export default router;
