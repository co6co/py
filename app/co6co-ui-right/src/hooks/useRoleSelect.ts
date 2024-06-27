import { ref } from 'vue';
import { showLoading, closeLoading, type ITreeSelect } from 'co6co';
import { association_service as ass_api } from '@/api/sys/role';

/**
 * 通过 roleId 获取菜单Tree
 */
export const useMenuTreeByRole = () => {
	const treeSelectData = ref<ITreeSelect[]>([]);
	const refresh = async (roleId: number) => {
		treeSelectData.value = [];
		showLoading();
		const res = await ass_api.get_association_svc(roleId, {});
		closeLoading();
		if (res.code == 0) treeSelectData.value = res.data;
	};
	return { treeSelectData, refresh };
};
