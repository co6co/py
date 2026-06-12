import { defineComponent, PropType, onMounted, VNode, watch } from 'vue';
import { ElSelect, ElOption } from 'element-plus';
import { useDictHook } from '@/hooks';
type ModelValueType = string | number | null | undefined;
type ValueUseField = 'id' | 'name' | 'flag' | 'value' | 'desc';
import { type DictSelectType } from '@/api/dict/dictType';
import { useModelWrapper } from 'co6co';

import { DictShowCategory } from '@/constants';
import { type IQueryDictSelectParam } from '@/api/dict/dict';
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
		dictTypeCode: {
			type: String as PropType<string>,
			required: false,
		},
		queryCategory: {
			type: Number as PropType<DictShowCategory>,
			default: DictShowCategory.NameValueFlag,
		},
		parentId: {
			type: Number as PropType<number>,
			default: undefined,
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
		
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (data: ModelValueType) => true,
		//change: (data: ModelValueType) => true,
	},

	setup(prop, ctx) {
		const { localValue, onChange } = useModelWrapper(prop, ctx);
		const stateHook = useDictHook.useDictSelect();
		
		const queryData = async ( dictTypeCode,parentId,queryCategory) => {
			if (!dictTypeCode) {
				console.warn("dictTypeCode 或 dictTypeId 不能为空同时为空");
				return
			}
			const param: IQueryDictSelectParam = {
				dictTypeCode: dictTypeCode,
				category: queryCategory,
				parentId: parentId,
			} 
			await stateHook.queryByCode(param);
		}
		onMounted(async () => {
			await queryData(prop.dictTypeCode,prop.parentId,prop.queryCategory)
		});
		const getName = (v) => {
			return stateHook.getName(String(v));
		};
		const flagIs = (value: string, flag: string) => {
			return stateHook.getFlag(value) == flag;
		};
		watch([() => prop.dictTypeCode,() => prop.parentId,() => prop.queryCategory], async (newValue, __) => {
			await await queryData(newValue[0],newValue[1],newValue[2])
		});
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
				<ElSelect v-model={localValue.value} onChange={onChange}
					placeholder={prop.placeholder} disabled={prop.disabled}
					clearable={prop.clearable} filterable={prop.filterable}>
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
