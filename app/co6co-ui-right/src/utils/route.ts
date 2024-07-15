import {
	IRouteData,
	getAllPermissionKeys,
	useRouteData,
} from '@/hooks/useRoute';
import { MenuCateCategory } from '@/hooks/useMenuSelect';
import { userSvc } from '@/api';
import { nextTick } from 'vue';
import {
	getToken,
	getRefreshToken,
	storeAuthonInfo,
	usePermissStore,
} from 'co6co';
import { Router } from 'vue-router';

export function registerRoute(ViewObjects, router: Router, bck?: () => void) {
	const { queryRouteData } = useRouteData();
	//console.info("to:",to,"from:",from)
	const permiss = usePermissStore();
	//console.info('route..query api...')
	queryRouteData((data: IRouteData[], e) => {
		//console.info('route..query api ed...')
		if (data && data.length > 0) {
			const list = getAllPermissionKeys(data);
			//console.info("所有权限字",list)
			permiss.set(list);
			addRoutes(ViewObjects, router, data); // 此处的menuList为上述中返回的数据
		} else {
			console.warn('获取路由数据失败或者为空', e);
		}
		if (bck) bck();
	});
}

// 动态添加路由
export function addRoutes(
	ViewObjects: any, //{path:view}
	router: Router,
	menu: IRouteData[]
) {
	menu.forEach((item) => {
		// 只将页面信息添加到路由中
		if (
			item.category == MenuCateCategory.VIEW ||
			item.category == MenuCateCategory.SubVIEW
		) {
			const component = ViewObjects[`${item.component}`]; //loadView[`../views${e.component}.vue`]
			//console.info("add route",item.name,item.code,"=>",item.component)
			component
				? router.addRoute('home', {
						name: item.code,
						path: item.url,
						meta: { title: item.name, permiss: item.permissionKey },
						component: component,
				  })
				: console.warn(`增加路由${item.name}找不到VIEW:${item.component}`);
		}
		if (item.children && item.children.length > 0)
			addRoutes(ViewObjects, router, item.children);
	});
}

//认证
export const validAuthenticate = (
	success: () => void,
	fail: () => void,
	ticket?: string
) => {
	const token = getToken();
	const refeshToken = getRefreshToken();
	do {
		if (!token) {
			if (ticket || refeshToken) {
				userSvc.ticket_svc(ticket || refeshToken).then((res) => {
					if (res.code == 0) {
						storeAuthonInfo(res.data);
						nextTick(() => success());
					} else fail();
				});
			} else {
				fail();
			}
			//next('/403');
		} else {
			success();
		}
	} while (false);
};
