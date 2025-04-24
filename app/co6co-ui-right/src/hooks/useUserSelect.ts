import { get_state_svc, get_category_svc } from '@/api/sys/user';
import { useEnum } from './base';
export const useState = () => {
	const { loadData, selectData, refresh, getName } = useEnum(get_state_svc);
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

	return { loadData, selectData, refresh, getName, getTagType };
};

export const useCategory = () => {
	const { loadData, selectData, refresh, getName } = useEnum(get_category_svc);
	return { loadData, selectData, refresh, getName };
};
