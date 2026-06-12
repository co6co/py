import { defineComponent, PropType, onMounted, VNode, watch, SetupContext } from 'vue';
import { ElSelect, ElOption } from 'element-plus';
import { useDictHook } from '@/hooks';
type ModelValueType = string | number | null | undefined;
type ValueUseField = 'id' | 'name' | 'flag' | 'value' | 'desc';

import { type DictSelectType } from '@/api/dict/dictType';
import { type IQueryDictSelectParam } from '@/api/dict/dict';
import { useModelWrapper } from 'co6co';

import { DictShowCategory } from '@/constants';
/**
 *  组件的封装
 *  1. 属性和事件的互传
 *  2. 插槽
 *  3. ref
 */
export default defineComponent({
	props: {
		placeholder: {
			type: String,
			default: '请选择',
		},

		queryParam: {
			type: Object as PropType<IQueryDictSelectParam>,
			required: false,
		},

		modelValue: {
			type: [String, Number, Object] as PropType<ModelValueType>, // 和下面没有任何区别
			//type: Object as PropType<ModelValueType>,
			//required: true,
			default: '',
		},
		valueUseField: {
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
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (data: ModelValueType) => true,
		//change: (data: ModelValueType) => true,
	},

	setup(prop, ctx: SetupContext) {
		//const { localValue, onChange } = useModelWrapper(prop, ctx);
		const stateHook = useDictHook.useDictSelect();
		// 声明 props
		//const props = defineProps(['name'])
		// 剩余透传属性 

		const { attrs } = ctx //,slots,emit ,expose

		//const attrs = useAttrs()
		//const slots = useSlots()
		const queryData = async (param: IQueryDictSelectParam) => {
			if (!param.dictTypeCode && !param.dictTypeId) {
				console.warn("dictTypeCode 或 dictTypeId 不能为空同时为空");
				return
			}
			if (param.category == undefined) param.category = DictShowCategory.NameValueFlag;
			await stateHook.queryByCode(param);
		}
		onMounted(async () => {
			if (prop.queryParam)
				await queryData(prop.queryParam)
		});
		const getName = (v) => {
			return stateHook.getName(String(v));
		};
		const flagIs = (value: string, flag: string) => {
			return stateHook.getFlag(value) == flag;
		};
		watch(() => prop.queryParam, async (newVal, __) => {
			if (!newVal) return;
			await queryData(newVal)
		}, { deep: true });
		const valueUse = (d: DictSelectType) => {
			let value: number | string | bigint = '';
			switch (prop.valueUseField) {
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
		const { localValue, onChange } = useModelWrapper(prop, ctx);
		const rander = (): VNode => {
			return (
				<ElSelect {...attrs} v-model={localValue.value} onChange={onChange}
					placeholder={prop.placeholder} disabled={prop.disabled}
					clearable={prop.clearable} filterable={prop.filterable}
				>
					{stateHook.selectData.value.map((d, index) => {
						return <ElOption key={index} label={d.name} value={valueUse(d)} />;
					})}

				</ElSelect>
			);
		};
		//真是方法 defineExpose 编译宏用于<script setup> 中的组件
		const exposedObj = {
			stateHook,
			getName,
			flagIs,
		}
		ctx.expose(exposedObj);
		//.d.ts 中的定义
		rander.stateHook = stateHook;
		rander.getName = getName;
		rander.flagIs = flagIs;
		return rander; // 顶层还有render方法，在这里返回 顶层【与seup同级就不行，不然这里应返回挂在在this对象上的对象】
	}, //end setup
});
