import { ref, onMounted } from 'vue';
import { IEnumSelect, ElTagType } from 'co6co';
import { get_state_svc } from '@/api/sys/user';

export const useState = () => {
	const selectData = ref<IEnumSelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await get_state_svc();
		if (res.code == 0) selectData.value = res.data;
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
	onMounted(() => {
		refresh();
	});
	return { selectData, refresh, getName, getTagType };
};
