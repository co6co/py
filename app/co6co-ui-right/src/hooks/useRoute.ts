import { ITreeSelect, removeAuthonInfo, getSession, setSession } from 'co6co';
import { view_route_svc, get_session_svc } from '@/api/view';
import { Storage, traverseTreeData, randomString } from 'co6co';
//确保你在 Vue 组件的 <script setup> 或 setup() 函数中使用 useRouter。
// 如果你需要在其他地方访问路由器，考虑通过参数传递路由器实例，或者在应用初始化时保存对它的引用。
import { useRouter } from 'vue-router';
import { ViewFeature } from '@/constants';
/**
 * VIEW 功能
 * 为控制页面按钮权限
 *
 * 需要调整 还需调整 useMethods
 * @returns
 */

/**
 * 获取当前路由
 * @returns 当前路由
 */
export const getCurrentRoute = () => {
	const router = useRouter();
	// 获取当前路由对象
	const currentRoute = router.currentRoute.value;
	return {
		currentRoute,
	};
};
/**
 * 获取指定视图权限字
 */
export default getCurrentRoute;
const getViewFeature = (pageUrl: string) => {
	const { getRouteData } = useRouteData();
	const data = getRouteData();
	let curremtPageRouteData: Partial<IRouteData> = {};
	if (data)
		traverseTreeData(data, (d) => {
			const item = d as IRouteData;
			//当为 subView 时有参数 /:id
			const index = item.url ? item.url.indexOf(':') : -1;
			if (
				pageUrl &&
				(item.url == pageUrl ||
					(index > -1 && pageUrl.indexOf(item.url.substring(0, index)) > -1))
			) {
				curremtPageRouteData = item;
				return true;
			}
		});
	return curremtPageRouteData.children;
};
/**
 * 获取按钮的权限字
 * 缓存中存有权限字
 * @param buttonType  user_add
 * @param pageUrl
 *
 * @returns
 */
const getPermissionKey = (
	buttonType: ViewFeature | string,
	pageUrl: string
) => {
	let permissionKey: string = '';
	const featureList = getViewFeature(pageUrl);

	if (featureList && featureList.length > 0) {
		const targetKey = `${buttonType}`;
		traverseTreeData(featureList, (d) => {
			const item = d as IRouteData;
			if (item.methods == targetKey) {
				permissionKey = item.permissionKey;
				return true;
			}
		});
	}
	if (permissionKey == null) {
		permissionKey = randomString(10);
		console.warn(`${pageUrl}.${buttonType}->permissionKey:${permissionKey}`);
	}
	return permissionKey;
};
/**
 * 权限相关
 * @returns
 */
export const usePermission = () => {
	const { currentRoute } = getCurrentRoute();
	const getCurrentViewFeature = () => {
		return getViewFeature(currentRoute.path);
	};
	/**
	 * 根据URLpath去找当前页面存储的权限字
	 * @param feature
	 * @returns
	 */
	const getPermissKey = (feature: ViewFeature) => {
		let result = getPermissionKey(feature, currentRoute.path);
		return result;
	};
	return { getPermissKey, getCurrentViewFeature };
};

export interface RouteItem {
	id: number;
	category: number;
	parentId: number;
	name: string;
	code: string;
	icon: string;
	url: string;
	component: string;
	methods: string;
	permissionKey: string;
}

/**
 * 左侧菜单数据类型
 */
export interface sideBarItem {
	icon: string;
	index: string;
	title: string;
	permiss: string;
	subs?: Array<sideBarItem>;
}

export type IRouteData = Omit<ITreeSelect, 'children'> &
	RouteItem & { children?: IRouteData[] };

/**
 * 获取配置的所有权限字
 */
export const getAllPermissionKeys = (data: IRouteData[] | null) => {
	if (!data) {
		const { getRouteData } = useRouteData();
		data = getRouteData();
	}
	let allKeys: string[] = [];
	if (data)
		traverseTreeData(data, (d) => {
			const item = d as IRouteData;
			if (item.permissionKey && !allKeys.includes(item.permissionKey))
				allKeys.push(item.permissionKey);
		});
	return allKeys;
};

/**
 * 通过根据后台登录用户获取菜单Tree
 */
export const useRouteData = () => {
	const Key = 'useRouteData';
	const storage = new Storage();
	const queryRouteData = (bck: (data: IRouteData[], msg?: string) => void) => {
		// 先获取session
		let session = getSession();
		if (!session) {
			get_session_svc().then((res) => {
				setSession(res.data.data, res.data.expiry);
			});
		}
		//看功能
		view_route_svc()
			.then((res) => {
				storage.set<IRouteData[]>(Key, res.data), bck(res.data);
			})
			.catch((e) => {
				removeAuthonInfo();
				if (e.message) bck([], e.message || '请求出错');
				else console.error(e), bck([], '请求出错');
			});
	};

	const getRouteData = () => {
		let result = storage.get<IRouteData[]>(Key);
		return result;
	};
	const clearRouteData = () => {
		storage.remove(Key);
	};
	/**
	 * 查询存储的路由
	 * @param filter 返回过滤
	 * @returns 仅返回第一条
	 */
	const queryRouteItem = (filter: (d: IRouteData) => boolean) => {
		const data = getRouteData();
		let result: { object?: RouteItem } = {};
		if (data) {
			traverseTreeData(data, (i) => {
				const d = i as IRouteData;
				if (filter(d)) {
					result.object = d;
					return true;
				}
			});
		}
		return result.object;
	};

	return { queryRouteData, getRouteData, queryRouteItem, clearRouteData };
};
