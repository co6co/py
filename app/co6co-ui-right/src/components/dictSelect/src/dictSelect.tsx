import { defineComponent, ref, watch, PropType, onMounted, VNode } from 'vue';
import { ElSelect, ElOption } from 'element-plus';
import { useDictHook } from '@/hooks';
type ModelValueType = string | number | null | undefined;
type ValueUseField = 'id' | 'name' | 'flag' | 'value' | 'desc';
import { type DictSelectType } from '@/api/dict/dictType';

import { DictShowCategory } from '@/constants';
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
		valueUseFiled: {
			type: String as PropType<ValueUseField>,
			default: 'value',
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
		queryCategory: {
			type: Number as PropType<DictShowCategory>,
			default: DictShowCategory.NameValueFlag,
		},
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (data: ModelValueType) => true,
		//change: (data: ModelValueType) => true,
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
			//ctx.emit('change', newValue);
			ctx.emit('update:modelValue', newValue);
		};
		const stateHook = useDictHook.useDictSelect();
		onMounted(async () => {
			await stateHook.queryByCode(prop.dictTypeCode, prop.queryCategory);
		});
		const getName = (v) => {
			return stateHook.getName(String(v));
		};
		const flagIs = (value: string, flag: string) => {
			return stateHook.getFlag(value) == flag;
		};
		const valueUse = (d: DictSelectType) => {
			let value: number | string | bigint = '';
			switch (prop.valueUseFiled) {
				case 'id':
					value = d.id;
					break;
				case 'name':
					value = d.name;
					break;
				case 'flag':
					value = d.flag;
					break;
				case 'value':
					value = d.value;
					break;
				case 'desc':
					value = d.desc;
					break;
				default:
			}
			return prop.isNumber ? Number(value) : value;
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
						return <ElOption key={index} label={d.name} value={valueUse(d)} />;
					})}
				</ElSelect>
			);
		};
		//真是方法
		ctx.expose({
			stateHook,
			getName,
			flagIs,
		});
		//.d.ts 中的定义
		rander.stateHook = stateHook;
		rander.getName = getName;
		rander.flagIs = flagIs;
		return rander;
	}, //end setup
});
