import { defineComponent, PropType } from 'vue';
import { IEnumSelect } from '@/constants';
import { ElSelect, ElOption } from 'element-plus';
import { useModelWrapper } from '@/hooks/useModelWrapper';
type ModelValueType = string | number | undefined;
export default defineComponent({
	name: 'EnumSelect',
	props: {
		data: {
			type: Array<IEnumSelect>,
			required: true,
		},
		modelValue: {
			type: [String, Number, undefined] as PropType<ModelValueType>,
			default: undefined,
		},
		placeholder: {
			type: String,
			default: '请选择',
		},
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (v: any) => true,
	},
	setup(prop, context) {
		//const localValue = ref<undefined | string | number>(prop.modelValue);
		//const onChange = () => {
		//	context.emit('update:modelValue', localValue.value);
		//};
		//watch(
		//	() => prop.modelValue,
		//	(v) => {
		//		localValue.value = v;
		//	}
		//);
		const { localValue, onChange } = useModelWrapper(prop, context);
		return () => {
			//可以写某些代码
			return (
				<ElSelect
					clearable
					style={context.attrs}
					class="mr10"
					v-model={localValue.value}
					onChange={onChange}
					placeholder={prop.placeholder}>
					{prop.data.map((d) => {
						return (
							<ElOption key={d.key} label={d.label} value={d.value}></ElOption>
						);
					})}
				</ElSelect>
			);
		};
	},
});
