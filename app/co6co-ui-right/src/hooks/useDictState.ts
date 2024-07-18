import { ref, onMounted } from 'vue';
import { IEnumSelect, ISelect } from 'co6co';
import { get_dict_state_svc } from '@/api/dict/dict';
import { get_select_svc, get_dict_select_svc } from '@/api/dict/dictType';

export const useState = () => {
	const selectData = ref<IEnumSelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await get_dict_state_svc();
		if (res.code == 0) selectData.value = res.data;
	};
	const getName = (value?: number) => {
		if (value != undefined)
			return selectData.value.find((m) => m.value == value)?.label;
		return '';
	};

	const getTagType = (value?: number) => {
		switch (value) {
			case 0:
				return 'danger';
			case 1:
				return 'success';
		}
	};
	onMounted(() => {
		refresh();
	});
	return { selectData, refresh, getName, getTagType };
};

export const useDictTypeSelect = () => {
	const selectData = ref<ISelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await get_select_svc();
		if (res.code == 0) selectData.value = res.data;
	};
	onMounted(() => {
		refresh();
	});
	return { selectData, refresh };
};
/**
 * 使用需要先查询
 * @returns
 */
export const useDictSelect = () => {
	const selectData = ref<ISelect[]>([]);
	const query = async (dictTypeId: number) => {
		selectData.value = [];
		const res = await get_dict_select_svc(dictTypeId);
		if (res.code == 0) selectData.value = res.data;
	};
	return { selectData, query };
};
