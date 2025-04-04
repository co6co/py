import { ref } from 'vue';
import { ISelect, ITreeSelect } from 'co6co';
import api from '@/api/sys/userGroup';
export default function () {
	const selectData = ref<ISelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await api.get_select_svc();
		selectData.value = res.data;
	};
	const getName = (value?: number) => {
		if (value) return selectData.value.find((m) => m.id == value)?.name;
		return '';
	};
	const loadData = refresh;
	return { loadData, selectData, refresh, getName };
}

export const useTree = (root?: number) => {
	const treeSelectData = ref<ITreeSelect[]>([]);
	const refresh = async () => {
		treeSelectData.value = [];
		const res = await api.get_select_tree_svc(root);
		treeSelectData.value = res.data;
	};
	const loadData = refresh;
	return { loadData, treeSelectData, refresh };
};
