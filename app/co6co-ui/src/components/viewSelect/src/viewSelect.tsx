import { ref, defineComponent, onMounted } from 'vue';
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
		'update:modelValue': (v: any) => true,
	},
	setup(prop, context) {
		const DATA = ref<undefined | string>(prop.modelValue);
		const store = getStoreInstance();
		const onChanged = () => {
			context.emit('update:modelValue', DATA.value);
		};
		onMounted(() => {});
		const vsolft = {
			default: () => {
				return (
					<ul>
						{Object.keys(store.views).map((key, index) => {
							return (
								<ElOption key={index} value={key}>
									{key}
								</ElOption>
							);
						})}
					</ul>
				);
			},

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
		};
		return () => {
			//可以写某些代码
			return (
				<ElSelect
					clearable
					filterable={true}
					style={context.attrs}
					class="iconList"
					v-model={DATA.value}
					onChange={onChanged}
					placeholder="请选择图标"
					v-slots={vsolft}></ElSelect>
			);
		};
	},
});
