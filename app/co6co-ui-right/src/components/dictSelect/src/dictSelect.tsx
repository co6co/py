import { defineComponent, PropType, onMounted, VNode } from 'vue';

import { ElSelect, ElOption } from 'element-plus';
import { useDictHook } from '@/hooks';
type ValueType = String | Number | Boolean;
export default defineComponent({
	name: 'dictSelect',
	props: {
		dictType: {
			type: String,
			required: true,
		},
		modelValue: {
			type: Object as PropType<ValueType>,
		},

		placeholder: {
			type: String,
			default: '请选择',
		},
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (data: string | number) => true,
	},
	setup: (prop, { emit, expose }) => {
		const stateHook = useDictHook.useDictSelect();
		onMounted(async () => {
			await stateHook.queryByCode(prop.dictType);
		});
		const onchange = (v) => {
			emit('update:modelValue', v);
		};
		const getName = (v) => {
			return stateHook.getName(v);
		};
		const render = (): VNode => {
			return (
				<ElSelect
					v-model={prop.modelValue}
					placeholder={prop.placeholder}
					onChange={(a) => {
						onchange(a);
					}}>
					{stateHook.selectData.value.map((d, index) => {
						return (
							<ElOption
								key={index}
								label={d.name}
								value={Number(d.value)}></ElOption>
						);
					})}
				</ElSelect>
			);
		};
		expose({
			stateHook,
			getName,
		});
		render.stateHook = stateHook;
		render.getName = getName;
		return render;
	}, //end setup
});
