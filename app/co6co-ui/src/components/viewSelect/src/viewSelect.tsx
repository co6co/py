import { ref, defineComponent } from 'vue';
import { ElSelect, ElOption, ElIcon, ElEmpty } from 'element-plus';
import { getStoreInstance } from '@/hooks';
import { Promotion } from '@element-plus/icons-vue';

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
	},
	setup(prop, context) {
		//const DATA = ref<undefined | string>(prop.modelValue);
		//console.info('1.view Setup...');
		const DATA = ref(prop.modelValue);
		const store = getStoreInstance();
		const onChanged = (newValue: undefined | string) => {
			DATA.value = newValue;
			context.emit('update:modelValue', newValue);
		};

		return () => {
			//可以写某些代码
			//console.info('2.rander...');
			//console.info('prop changeed...');
			DATA.value = prop.modelValue;
			return (
				<ElSelect
					clearable
					filterable={true}
					style={context.attrs}
					class="iconList"
					v-model={DATA.value}
					onChange={onChanged}
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
