import { ref, Component } from 'vue';
import { useRouteData, type RouteItem } from '@/hooks/useRoute';
import {
	getViewPath /*views*/,
	getStoreInstance,
	isIView,
	isViewComponent,
} from 'co6co';
import { Router } from 'vue-router';
import { replaceRouteParams } from '@/utils';

/**
 * 获取子视图Path
 * @param mainViewName 必须再 views 中
 */
export const useViewData = (
	viewName: string,
	getMainViewPath?: (string) => string
) => {
	const subViewPath = ref('');
	const viewDataRef = ref<RouteItem>();
	const { queryRouteItem } = useRouteData();
	if (!getMainViewPath) getMainViewPath = getViewPath;
	const componentName = getMainViewPath(viewName);
	const routeItem = queryRouteItem((d) => {
		return d.component == componentName;
	});
	if (routeItem) {
		subViewPath.value = routeItem.url;
		viewDataRef.value = routeItem;
	}
	return { subViewPath, viewDataRef, replaceRouteParams };
};

export const goToPath = (
	router: Router,
	path,
	param: Record<string, string>
) => {
	//const router = useRouter(); //在这里无法获得router
	const tempPath = replaceRouteParams(path, param);
	//1.  router.push('/home')
	//2.  router.push({ path: '/user/johnny' }) 				 //提供了 path，params 将会被忽略
	//3.  router.push({ name: 'user', params: { userId: 123 }})  //命名路由的名字。使用命名路由时，你可以通过 params 传递参数，而不会被忽略
	router.push({ path: tempPath });
};
export const goToNameRoute = (
	router: Router,
	param: {
		name: string;
		//2022-8-22 这次更新后,无法获取该参数
		params?: Record<string, string>;
		state?: Record<string, any>;
	}
) => {
	//const router = useRouter(); //在这里无法获得router
	router.push(param);
};
/**
 * 获取该 试图下面的功能
 * @param viewPath 组件地址
 * @returns 功能对象 {add:'add',other:{value:'other',text:'功能1'}}
 */
export const useViewFeature = async (viewPath: string) => {
	const store = getStoreInstance();
	const loader = store.views[viewPath];
	if (isIView(loader)) {
		return loader.features;
	} else if (isViewComponent(loader) && typeof loader == 'function') {
		const loader2 = loader as () => Promise<{
			default: Component;
			features?: object;
		}>;
		const com = await loader2();
		const component = com.default;
		return com.features;
	}
};
