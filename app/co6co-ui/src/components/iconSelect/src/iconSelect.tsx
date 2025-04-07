import { defineComponent, resolveComponent, h } from 'vue';
import { ElSelect, ElOption, ElIcon, ElEmpty } from 'element-plus';
import * as icon from '@element-plus/icons-vue';
import iconStyle from '@/assets/css/eciconselect.module.less';
import { useModelWrapper } from '@/hooks/useModelWrapper';

export default defineComponent({
	name: 'EcIconSelect',
	props: {
		modelValue: {
			type: String,
		},
		placeholder: {
			type: String,
			default: '请选择',
		},
	},
	emits: {
		//@ts-ignore
		'update:modelValue': (v: string | undefined) => true,
	},
	setup(prop, context) {
		//const localValue = ref<undefined | string>(prop.modelValue);
		//const onChange = (newValue: undefined | string) => {
		//	localValue.value = newValue;
		//	context.emit('update:modelValue', newValue);
		//};
		//watch(
		//	() => prop.modelValue,
		//	(v) => {
		//		localValue.value = v;
		//	}
		//);

		const { localValue, onChange } = useModelWrapper(prop, context);
		const vsolft = {
			default: () => {
				return (
					<ul class={iconStyle.iconList}>
						{Object.keys(icon).map((key, index) => {
							return (
								<ElOption key={index} class={iconStyle.icon_item} value={key}>
									<ElIcon>{h(resolveComponent(key))}</ElIcon>
								</ElOption>
							);
						})}
					</ul>
				);
			},

			prefix: () => {
				return localValue.value ? (
					<ElIcon color="#c6c6c6">
						{h(resolveComponent(localValue.value))}
					</ElIcon>
				) : (
					<>空</>
				);
			},
			empty: () => {
				return <ElEmpty></ElEmpty>;
			},
		};
		return () => {
			//可以写某些代码
			return (
				<ElSelect
					clearable
					filterable={true}
					style={context.attrs}
					class="iconList"
					v-model={localValue.value}
					onChange={onChange}
					placeholder="请选择图标"
					v-slots={vsolft}
				/>
			);
		};
	},
});
