import Home from '../views/home.vue';
import { getToken, removeToken } from '../utils/auth';   
import { createRouter, createWebHistory,createWebHashHistory } from 'vue-router';
import { usePermissStore } from '../store/permiss';
 

const router = createRouter({ 
    history: createWebHistory(import.meta.env.VITE_UI_PATH), //有二級路徑需要配置，
    routes:  [ 
        /**
         * 
         */
        {
            path: '/',
            redirect: '/processAudit',
        },
        {
            path: '/home',
            name: 'Home',
            component: Home,
            children: [ 
                {
                    path: '/usermgr',
                    name: 'userTable',
                    meta: {
                        title: '用户管理',
                        permiss: 'admin',
                    },
                    component: () => import(  '../sysViews/userTable.vue'),
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
                    path: '/processRecord',
                    name: 'processRecord',
                    meta: {
                        title: '记录查询',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/processRecord.vue'),
                },
                {
                    path: '/processAudit',
                    name: 'processAudit',
                    meta: {
                        title: '记录审核',
                        permiss: 'user',
                    },
                    component: () => import( '../views/processAudit.vue'),
                },
                {
                    path: '/taskTable',
                    name: 'taskTable',
                    meta: {
                        title: '任务列表',
                        permiss: '2',
                    },
                    component: () => import( '../views/taskTable.vue'),
                },
                {
                    path: '/groupTable',
                    name: 'groupTable',
                    meta: {
                        title: '分组信息',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/groupTable.vue'),
                },
                {
                    path: '/labelTable',
                    name: 'labelTable',
                    meta: {
                        title: '标签管理',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/labelTable.vue'),
                },
                {
                    path: '/processLable',
                    name: 'processLable',
                    meta: {
                        title: '数据管理',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/processLable.vue'),
                }, 

                {
                    path: '/userBoat',
                    name: 'userBoat',
                    meta: {
                        title: '船关联',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/userboat.vue'),
                }, 
                {
                    path: '/userJobs',
                    name: 'userJobs',
                    meta: {
                        title: '审核任务',
                        permiss: 'user',
                    },
                    component: () => import( '../views/userJobs.vue'),
                }, 
                {
                    path: '/userJobsDetail/:id',
                    name: 'userJobsDetail',
                    meta: {
                        title: '审核任务详情',
                        permiss: 'user',
                    },
                    component: () => import( '../views/userJobsDetail.vue'),
                }, 
                {
                    path: '/ruleTable',
                    name: 'ruleTable',
                    meta: {
                        title: '规则管理',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/ruleTable.vue'),
                }, 
                {
                    path: '/auditSetting',
                    name: 'auditSetting',
                    meta: {
                        title: '审核配置',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/auditSetting.vue'),
                }, 
                {
                    path: '/groupTree',
                    name: 'groupTree',
                    meta: {
                        title: '船优先级',
                        permiss: 'admin',
                    },
                    component: () => import( '../views/groupTree.vue'),
                }, 

                /**权限 */
                {
                    path: '/userGroup',
                    name: 'userGroup',
                    meta: {
                        title: '用户组',
                        permiss: 'admin',
                    },
                    component: () => import( '../sysViews/userGroupTreeView.vue'),
                }, 
                {
                    path: '/menu',
                    name: 'menu',
                    meta: {
                        title: '菜单',
                        permiss: 'admin',
                    },
                    component: () => import( '../sysViews/menuTreeView.vue'),
                }, 
                {
                    path: '/role',
                    name: 'role',
                    meta: {
                        title: '角色',
                        permiss: 'admin',
                    },
                    component: () => import( '../sysViews/roleView.vue'),
                }, 
            ],
        } ,
        // 以上需要动态
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
    ]
  })  
  

router.beforeEach((to, from, next) => {
	document.title = `${to.meta.title} `;
	//const role = storeage.get('username');
	const permiss = usePermissStore();
	const token = getToken(); 
	if (!token && to.path !== '/login') {
		next('/login');
	} else if (to.meta.permiss && !permiss.key.includes(to.meta.permiss)) {
		// 如果没有权限，则进入403
		removeToken()
		next('/403');
	} else {
       // console.info(to.path)//查找有空白的情况
       // console.info("to:",to,"from:",from)
		next();
	}
});
export default router;
