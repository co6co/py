import { IEnumSvc } from '@/constants';
import { ref } from 'vue';
import { IEnumSelect } from 'co6co';
export const useEnum = (svc: IEnumSvc) => {
	const selectData = ref<IEnumSelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await svc();
		selectData.value = res.data;
	};
	const getName = (value?: number) => {
		if (value != undefined)
			return selectData.value.find((m) => m.value == value)?.label;
		return '';
	};

	const loadData = refresh;
	return { loadData, selectData, refresh, getName };
};
