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

export const usePageFeature = () => {
	let index = -1;
	const list: Array<IEnumSelect> = [];
	Object.keys(ViewFeature).forEach((key) => {
		console.log(key, ViewFeature[key as keyof typeof ViewFeature]);
		list.push({
			uid: ++index,
			key: key,
			label: ViewFeatureDesc.getDesc(key),
			value: key,
		});
	});

	const selectData = ref<IEnumSelect[]>(list);
	const getName = (value?: number) => {
		if (value) return selectData.value.find((m) => m.value == value)?.label;
		return '';
	};
	return { selectData, getName };
};
