import { ref, onMounted } from 'vue';
import { IEnumSelect, ISelect } from 'co6co';
import {
	get_dict_state_svc,
	get_dict_select_svc as get_dict_select_by_code_svc,
} from '@/api/dict/dict';
import {
	get_select_svc,
	get_dict_select_svc,
	type DictSelectType,
} from '@/api/dict/dictType';
/**
 * 状态 api
 * @returns 状态[]
 */
export const useState = () => {
	const selectData = ref<IEnumSelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await get_dict_state_svc();
		selectData.value = res.data;
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
/**
 * 字典类型API
 * @returns 字典类型 Select[]
 */

export const useDictTypeSelect = () => {
	const selectData = ref<ISelect[]>([]);
	const refresh = async () => {
		selectData.value = [];
		const res = await get_select_svc();
		selectData.value = res.data;
	};
	onMounted(() => {
		refresh();
	});
	return { selectData, refresh };
};
/**
 * 使用需要先查询
 * @returns 字典Select[]
 */
export const useDictSelect = () => {
	const selectData = ref<DictSelectType[]>([]);
	const query = async (dictTypeId: number) => {
		selectData.value = [];
		const res = await get_dict_select_svc(dictTypeId);
		selectData.value = res.data;
	};
	const queryByCode = async (dictTypeCode: string) => {
		selectData.value = [];
		const res = await get_dict_select_by_code_svc(dictTypeCode);
		selectData.value = res.data;
	};
	const _checkSelectDataValid = () => {
		return selectData.value && selectData.value.length > 0;
	};
	const getName = (value: string) => {
		//selectData.value.filter((m) => m.value == value);
		//console.info('dictType', selectData.value);
		if (_checkSelectDataValid())
			return selectData.value.find((m) => m.value == value)?.name;
	};
	const getFlag = (value: string) => {
		//selectData.value.filter((m) => m.value == value);
		if (_checkSelectDataValid())
			return selectData.value.find((m) => m.value == value)?.flag;
	};
	const getDesc = (value: string) => {
		//selectData.value.filter((m) => m.value == value);
		if (_checkSelectDataValid())
			return selectData.value.find((m) => m.value == value)?.desc;
	};
	return { selectData, query, queryByCode, getName, getFlag, getDesc };
};
