import { ref } from 'vue';
import { useRouteData, type RouteItem } from '@/hooks/useRoute';

import { ViewFeature, ViewSubFeatures } from '@/constants';
import {
	getViewPath /*views*/,
	getStoreInstance,
	isTsxView,
	isFuncView,
} from 'co6co';
import { Router } from 'vue-router';
import { replaceRouteParams } from '@/utils';
import { MenuCateCategory } from '@/hooks/useMenuSelect';

/**
 * 获取子视图Path
 * @param mainViewName 必须再 views 中
 * @param moduleName 模块名,不是getViewPath 所在模块必须提供模块名，主模块使用 ".."  作为模块名
 */
export const useViewData = (viewName: string, moduleName?: string) => {
	const viewPath = ref('');
	const viewDataRef = ref<RouteItem>();
	const { queryRouteItem } = useRouteData();
	let componentName = ""
	if (moduleName)
		componentName = getViewPath(viewName, moduleName); 
	else
		componentName = getViewPath(viewName); 
	const routeItem = queryRouteItem((d) => {
		//return componentName==d.component; //主模块没有后缀
		if(!d.component){
			return false
		}
		
		return d.component.indexOf(componentName)>-1;
		//return componentName.includes(d.component);
	});
	if (routeItem) {
		viewPath.value = routeItem.url;
		viewDataRef.value = routeItem;
	}
	return { viewPath, viewDataRef, replaceRouteParams };
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
 * 获取该 视图 功能
 * @param viewPath 组件地址
 * @returns 功能对象 {add:'add',other:{value:'other',text:'功能1'}}
 */
export const queryViewFeature = async (
	viewPath: string,
	category: MenuCateCategory
) => {
	switch (category) {
		case MenuCateCategory.Button:
			const store = getStoreInstance();
			const viewComponent = store.views[viewPath];
			if (isTsxView(viewComponent)) {
				return viewComponent.features;
			} else if (isFuncView(viewComponent)) {
				const com = await viewComponent();
				//const component = com.default;
				return com.features!;
			} else {
				//默认的 组件 ，没有 features 属性
				return ViewFeature;
			}
		case MenuCateCategory.SubVIEW:
			return ViewSubFeatures;
		default:
			return {};
	}
};
