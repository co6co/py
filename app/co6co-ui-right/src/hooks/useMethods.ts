import { ref } from 'vue';
import { IEnumSelect } from 'co6co';
import { ViewFeature } from './useRoute';
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
	const selectData = ref<IEnumSelect[]>([
		{
			uid: ++index,
			key: ViewFeature.view,
			label: '查看',
			value: ViewFeature.view,
		},
		{
			uid: ++index,
			key: ViewFeature.get,
			label: '获取',
			value: ViewFeature.get,
		},
		{
			uid: ++index,
			key: ViewFeature.downloads,
			label: '下载多个',
			value: ViewFeature.downloads,
		},
		{
			uid: ++index,
			key: ViewFeature.download,
			label: '下载',
			value: ViewFeature.download,
		},
		{
			uid: ++index,
			key: ViewFeature.upload,
			label: '上传',
			value: ViewFeature.upload,
		},
		{
			uid: ++index,
			key: ViewFeature.add,
			label: '增加',
			value: ViewFeature.add,
		},
		{
			uid: ++index,
			key: ViewFeature.edit,
			label: '编辑',
			value: ViewFeature.edit,
		},
		{
			uid: ++index,
			key: ViewFeature.check,
			label: '检查',
			value: ViewFeature.check,
		},
		{
			uid: ++index,
			key: ViewFeature.setting,
			label: '设置',
			value: ViewFeature.setting,
		},
		{
			uid: ++index,
			key: ViewFeature.del,
			label: '删除',
			value: ViewFeature.del,
		},
		{
			uid: ++index,
			key: ViewFeature.associated,
			label: '关联',
			value: ViewFeature.associated,
		},
		{
			uid: ++index,
			key: ViewFeature.reset,
			label: '重置',
			value: ViewFeature.reset,
		},
		{
			uid: ++index,
			key: ViewFeature.push,
			label: '推送',
			value: ViewFeature.push,
		},
		{
			uid: ++index,
			key: ViewFeature.effective,
			label: '使生效',
			value: ViewFeature.effective,
		},
		{
			uid: ++index,
			key: ViewFeature.settingName,
			label: '设置名称',
			value: ViewFeature.settingName,
		},
		{
			uid: ++index,
			key: ViewFeature.settingNo,
			label: '设置编号',
			value: ViewFeature.settingNo,
		},
		{
			uid: ++index,
			key: ViewFeature.settingPriority,
			label: '设置优先级',
			value: ViewFeature.settingPriority,
		},
	]);

	const getName = (value?: number) => {
		if (value) return selectData.value.find((m) => m.value == value)?.label;
		return '';
	};
	return { selectData, getName };
};
