import { ref, defineComponent, type PropType } from 'vue';

//@ts-ignore
import {
	ElForm,
	ElMessage,
	type FormInstance,
	type FormRules,
} from 'element-plus';

export interface IFormDataType {
	visible: boolean;
	title?: string;
	loading: boolean;
}

export default defineComponent({
	name: 'EcForm',
	props: {
		rules: {
			type: Object as PropType<FormRules>,
			required: false,
		},
		model: {
			type: Object as PropType<Record<string, any>>, //as PropType<ObjectConstructor>,
			required: true,
		},
		labelWidth: {
			type: Number, //as PropType<ObjectConstructor>,
			default: 150,
		},
	},

	emits: {
		//@ts-ignore
		error: (msg: string, ValidateFieldsError) => true,
		submit: () => true,
	},
	setup(prop, { attrs, slots, emit, expose }) {
		const formRef = ref<FormInstance>();
		const _checkData = (
			instance: FormInstance | undefined,
			successBck?: () => void,
			validateBefore?: () => void
		) => {
			if (!instance) {
				ElMessage.error('表单对象为空！');
				emit('error', '表单对象为空！', {});
				return false;
			}
			if (validateBefore) validateBefore();

			instance.validate((value, invalidFields) => {
				if (!value) {
					ElMessage.error('请检查输入的数据！');
					emit('error', '请检查输入的数据！', invalidFields);
					return Promise.reject('valid Form Error');
				}
				//提交数据
				if (successBck) successBck();
				emit('submit');
			});
		};
		const validate = (successBck?: () => void, validateBefore?: () => void) => {
			_checkData(formRef.value, successBck, validateBefore);
		};
		expose({
			formInstance: formRef,
			validate: validate,
		});
		const render = () => {
			//可以写某些代码
			return (
				<ElForm
					style={attrs}
					labelWidth={prop.labelWidth}
					ref={formRef}
					rules={prop.rules}
					model={prop.model}>
					{slots.default ? slots.default() : null}
				</ElForm>
			);
		};

		render.formInstance = formRef;
		render.validate = validate;
		return render;
	},
});
