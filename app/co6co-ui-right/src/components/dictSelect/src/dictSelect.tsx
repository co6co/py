import {
	defineComponent,
	ref,
	watch,
	PropType,
	onMounted,
	VNode,
	computed,
} from 'vue';
import { definePropType } from 'co6co';
import { ElSelect, ElOption } from 'element-plus';
import { useDictHook } from '@/hooks';
type ModelValueType = string | number | null | undefined;
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
			type: [String, Number, Object] as PropType<ModelValueType>, // 和下面没有任何区别
			//type: Object as PropType<ModelValueType>,
			//required: true,
			default: '',
		},
		isNumber: {
			type: Boolean as PropType<Boolean>,
			default: false,
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
			return stateHook.getName(String(v));
		};

		const rander = (): VNode => {
			return (
				<ElSelect
					disabled={prop.disabled}
					filterable={prop.filterable}
					clearable={prop.clearable}
					v-model={localValue.value}
					placeholder={prop.placeholder}
					onChange={onChange}>
					{stateHook.selectData.value.map((d, index) => {
						return (
							<ElOption
								key={index}
								label={d.name}
								value={prop.isNumber ? Number(d.value) : d.value}
							/>
						);
					})}
				</ElSelect>
			);
		};
		//真是方法
		ctx.expose({
			stateHook,
			getName,
		});
		//.d.ts 中的定义
		rander.stateHook = stateHook;
		rander.getName = getName;
		return rander;
	}, //end setup
});
