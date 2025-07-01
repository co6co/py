import { ref } from 'vue';
import { IEnumSelect } from 'co6co';
import { ViewFeature, ViewFeatureDesc } from '@/constants';
/**
 * HTTP Methods
 * @returns
 */
export default () => {
	const selectData = ref<IEnumSelect[]>([
		{ uid: 0, key: 'GET', label: 'all操作', value: 'ALL' },
		{ uid: 1, key: 'GET', label: 'get操作', value: 'GET' },
		{ uid: 2, key: 'POST', label: 'post操作', value: 'POST' },
		{ uid: 3, key: 'PUT', label: 'put操作', value: 'PUT' },
		{ uid: 4, key: 'PATCH', label: 'patch操作', value: 'PATCH' },
		{ uid: 5, key: 'DELETE', label: 'delete操作', value: 'DELETE' },
	]);

	const getName = (value?: number) => {
		if (value) return selectData.value.find((m) => m.value == value)?.label;
		return '';
	};
	return { selectData, getName };
};

/**
 *
 * @param pageFeature 页面字段 {get:"get",add:"add" sched:{value:"sched",text:"调度表"}}
 * @returns
 */
export const useFeatureSelect = (pageFeature?: object) => {
	let index = -1;
	const list: Array<IEnumSelect> = [];
	pageFeature = pageFeature ?? ViewFeature;
	Object.keys(pageFeature).forEach((key) => {
		//console.log(key, ViewFeature[key as keyof typeof ViewFeature]);
		const value = ViewFeatureDesc.getValue(key, pageFeature);
		list.push({
			uid: ++index,
			key: value,
			label: ViewFeatureDesc.getDesc(key, pageFeature),
			value: value,
		});
	});

	const selectData = ref<IEnumSelect[]>(list);
	const getName = (value?: number) => {
		if (value) return selectData.value.find((m) => m.value == value)?.label;
		return '';
	};
	return { selectData, getName };
};
