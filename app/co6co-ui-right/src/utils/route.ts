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
	getPermissStoreInstance,
	getStoreInstance,
} from 'co6co';
import { Router } from 'vue-router';
/**

 * @param route  带冒号的路径: /abc/:id/:abc
 * @param params 对象
 * @returns 
 * 
 *  const route = "/abc/:id/:abc";
    const params = { id: '1', abc: 'happy' }; 
	输出: /abc/1/happy
 */
export function replaceRouteParams(
	route: string,
	params: Record<string, string>
): string {
	return route.replace(/:(\w+)/g, (_, paramName) => {
		const paramValue = params[paramName];
		if (paramValue === undefined) {
			throw new Error(
				`Parameter '${paramName}' is missing in the provided parameters.`
			);
		}
		return paramValue;
	});
}

export function registerRoute(router: Router, bck?: (msg?: string) => void) {
	const { queryRouteData } = useRouteData();
	//console.info("to:",to,"from:",from)
	const permiss = getPermissStoreInstance();
	//console.info('route..query api...')
	const ViewObjects = getStoreInstance().views;
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
		if (bck) bck(e);
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
				userSvc
					.ticket_svc(ticket || refeshToken)
					.then((res) => {
						storeAuthonInfo(res.data);
						nextTick(() => success());
					})
					.catch((e) => {
						fail();
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
