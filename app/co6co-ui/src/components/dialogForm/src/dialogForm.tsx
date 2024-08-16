import { defineComponent, inject, ref, VNode } from 'vue';
import { ElButton, type FormRules } from 'element-plus';
import { Form, Dialog } from '@/components';
import type { DialogInstance, FormInstance } from '@/components';

import type { PropType } from 'vue';

//Omit、Pick、Partial、Required
/*
export interface Item {
	id: number;
	name: string;
	postionInfo: string;
	deviceCode: string;
	deviceDesc: string;
	createTime: string;
	updateTime: string;
}*/
//Omit、Pick、Partial、Required
//export type FormItem = Omit<Item, 'id' | 'createTime' | 'updateTime'>;

export default defineComponent({
	name: 'EcdiaglogForm',
	props: {
		title: {
			type: String,
		},
		model: {
			type: Object as PropType<Record<string, any>>, //as PropType<ObjectConstructor>,
		},
		rules: {
			type: Object as PropType<FormRules>,
		},
		labelWidth: {
			type: Number, //as PropType<ObjectConstructor>,
			default: 150,
		},
	},
	emits: {
		error: (msg: string) => true,
		submit: () => true,
	},
	setup(prop, ctx) {
		const dialogRef = ref<DialogInstance>();
		const formInstance = ref<FormInstance>();
		const setDiaglogVisible = (show: boolean) => {
			if (dialogRef.value) {
				dialogRef.value.data.visible = show;
			}
		};
		const slots = {
			buttons: () => (
				<ElButton onClick={() => formInstance.value?.validate()}>保存</ElButton>
			),
		};
		// const data: Object = inject('formData') || {}
		const data =
			prop.model || ((inject('formData') || {}) as Record<string, any>);
		const openDialog = () => {
			if (dialogRef.value) {
				dialogRef.value.data.title = prop.title || '弹出框';
				setDiaglogVisible(true);
			}
		};
		const closeDialog = () => {
			setDiaglogVisible(false);
		};

		const rander = (): VNode => {
			return (
				<Dialog
					title={prop.title}
					style={ctx.attrs}
					ref={dialogRef}
					v-slots={ctx.slots.buttons ? { buttons: ctx.slots.buttons } : slots}>
					<Form
						style="0 45px 0 0"
						labelWidth={prop.labelWidth}
						v-slots={ctx.slots.default}
						ref={formInstance}
						rules={prop.rules}
						model={data}
						onSubmit={() => ctx.emit('submit')}
						onError={(e) => ctx.emit('error', e)}
					/>
				</Dialog>
			);
		};
		//暴露方法给父组件
		const validate = (success: () => void, validateBefore?: () => void) => {
			formInstance.value?.validate(success, validateBefore);
		};
		ctx.expose({
			openDialog,
			closeDialog,
			formInstance,
			validate,
		});

		rander.openDialog = openDialog;
		rander.closeDialog = closeDialog;
		rander.formInstance = formInstance;
		rander.validate = validate;
		return rander;
	}, //end setup
});
