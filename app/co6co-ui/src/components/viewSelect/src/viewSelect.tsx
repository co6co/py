import { defineComponent } from 'vue';
import { ElSelect, ElOption, ElIcon, ElEmpty } from 'element-plus';
import { getStoreInstance, useModelWrapper } from '@/hooks';
import { Promotion } from '@element-plus/icons-vue';
import { TIView } from '@/constants/types';

export default defineComponent({
	name: 'ViewSelect',
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
		'update:modelValue': (v: undefined | string) => true,
		//定义 事件不要加 on ，vue 编译时会自动加上，这个问题花了1h才发现
		change: (componentPath: string, component: TIView) => true,
	},
	setup(prop, context) {
		//const DATA = ref<undefined | string>(prop.modelValue);
		//console.info('1.view Setup...');
		//const localValue = ref(prop.modelValue);
		//const onChange = (newValue: undefined | string) => {
		//	localValue.value = newValue;
		//	context.emit('update:modelValue', newValue);
		//};
		const store = getStoreInstance();
		const { localValue, onChange } = useModelWrapper(prop, context);
		const onChange2 = (newValue) => {
			const data = store.views[newValue];
			context.emit('change', newValue, data);
			onChange(newValue);
		};
		return () => {
			localValue.value = prop.modelValue;
			return (
				<ElSelect
					clearable
					filterable={true}
					style={context.attrs}
					class="iconList"
					v-model={localValue.value}
					onChange={onChange2}
					placeholder="请选择视图">
					{{
						default: () => (
							<>
								{Object.keys(store.views).map((key, index) => {
									return <ElOption key={index} value={key} label={key} />;
								})}
							</>
						),

						prefix: () => {
							return (
								<ElIcon>
									<Promotion />
								</ElIcon>
							);
						},
						empty: () => {
							return <ElEmpty></ElEmpty>;
						},
					}}
				</ElSelect>
			);
		};
	},
});
