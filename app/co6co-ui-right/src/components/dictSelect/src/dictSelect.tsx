import { defineComponent, ref, watch, PropType, onMounted, VNode } from 'vue';
import { ElSelect, ElOption } from 'element-plus';
import { useDictHook } from '@/hooks';
type ModelValueType = string | number;
export default defineComponent({
	props: {
		placeholder: {
			type: String,
			default: '请选择',
		},
		dictTypeCode: {
			type: String as PropType<string>,
			required: true,
		},
		modelValue: {
			type: [String, Number] as PropType<ModelValueType>,
			required: true,
		},
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (data: string | number) => true,
	},
	setup(prop, ctx) {
		//存储本地值
		const localValue = ref(prop.modelValue);
		// 监听 modelValue 的变化 更新本地值
		watch(
			() => prop.modelValue,
			(newValue) => {
				localValue.value = newValue;
			}
		);
		const onChange = (newValue: ModelValueType) => {
			localValue.value = newValue;
			ctx.emit('update:modelValue', newValue);
		};
		const stateHook = useDictHook.useDictSelect();
		onMounted(async () => {
			await stateHook.queryByCode(prop.dictTypeCode);
		});
		const getName = (v) => {
			return stateHook.getName(v);
		};
		const rander = (): VNode => {
			return (
				<ElSelect
					v-model={localValue.value}
					placeholder={prop.placeholder}
					onChange={onChange}>
					{stateHook.selectData.value.map((d, index) => {
						return <ElOption key={index} label={d.name} value={d.value} />;
					})}
				</ElSelect>
			);
		};

		ctx.expose({
			stateHook,
			getName,
		});
		rander.stateHook = stateHook;
		rander.getName = getName;
		return rander;
	}, //end setup
});
