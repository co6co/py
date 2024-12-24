import { ref } from 'vue';
import { IEnumSelect } from 'co6co';
import { get_state_svc } from '@/api/sys/user';

export const useState = () => {
	const selectData = ref<IEnumSelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await get_state_svc();
		selectData.value = res.data;
	};
	const getName = (value?: number) => {
		if (value != undefined)
			return selectData.value.find((m) => m.value == value)?.label;
		return '';
	};

	const getTagType = (value?: number) => {
		if (value != undefined) {
			switch (value) {
				case 0:
					return 'success';
				case 1:
					return 'danger';
				case 2:
					return 'warning';
			}
		}
		return 'info';
	};

	const loadData = refresh;
	return { loadData, selectData, refresh, getName, getTagType };
};
