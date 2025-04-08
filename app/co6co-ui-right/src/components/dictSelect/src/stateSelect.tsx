import { defineComponent, onMounted, PropType, VNode } from 'vue';
import { ElSelect, ElOption } from 'element-plus';

import { useState } from '@/hooks/useDictState';
import { useModelWrapper } from 'co6co';
type ModelValueType = string | number | null | undefined;
export default defineComponent({
	props: {
		placeholder: {
			type: String,
			default: '请选择',
		},
		modelValue: {
			type: [String, Number, Object] as PropType<ModelValueType>, // 和下面没有任何区别
			//type: Object as PropType<ModelValueType>,
			//required: true,
			default: '',
		},

		disabled: {
			type: Boolean as PropType<Boolean>,
			default: false,
		},
		clearable: {
			type: Boolean as PropType<Boolean>,
			default: true,
		},
		filterable: {
			type: Boolean as PropType<Boolean>,
			default: true,
		},
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (data: ModelValueType) => true,
	},
	setup(prop, ctx) {
		const { localValue, onChange } = useModelWrapper(prop, ctx);
		const { loadData, selectData, getName, getTagType } = useState();
		onMounted(async () => {
			await loadData();
		});

		const rander = (): VNode => {
			return (
				<ElSelect
					disabled={prop.disabled}
					filterable={prop.filterable}
					clearable={prop.clearable}
					v-model={localValue.value}
					placeholder={prop.placeholder}
					onChange={onChange}>
					{selectData.value.map((d, index) => {
						return <ElOption key={index} label={d.label} value={d.value} />;
					})}
				</ElSelect>
			);
		};
		//真是方法
		ctx.expose({
			getName,
			getTagType,
			selectData,
		});
		//.d.ts 中的定义
		rander.getName = getName;
		rander.getTagType = getTagType;
		rander.selectData = selectData;
		return rander;
	}, //end setup
});
