import { defineComponent, ref, reactive, provide, VNode } from 'vue';
import type { InjectionKey } from 'vue';
import {
	DialogForm,
	FormOperation,
	showLoading,
	closeLoading,
	type DialogFormInstance,
	type FormData,
} from 'co6co';

import * as api_type from 'co6co';
import { dictSvc as svc } from '@/api/dict';

import {
	ElRow,
	ElCol,
	ElButton,
	ElFormItem,
	ElInput,
	ElMessage,
	type FormRules,
	ElInputNumber,
} from 'element-plus';

import { StateSelect } from '@/components/dictSelect';
export interface Item extends api_type.FormItemBase {
	id: number;
	dictTypeId?: number;
	name?: string;
	value?: string;
	flag?: string;
	state?: number;
	desc?: string;
	order: number;
}
//Omit、Pick、Partial、Required
export type FormItem = Omit<
	Item,
	'id' | 'createUser' | 'updateUser' | 'createTime' | 'updateTime'
>;
export default defineComponent({
	props: {
		title: {
			type: String,
		},
		labelWidth: {
			type: Number, //as PropType<ObjectConstructor>,
			default: 110,
		},
	},
	emits: {
		//@ts-ignore
		saved: (data: any) => true,
	},
	setup(prop, ctx) {
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,
			id: 0,
			fromData: { order: 0 },
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const init_data = (
			oper: FormOperation,
			dictTypeId: number,
			item?: Item
		) => {
			DATA.operation = oper;
			switch (oper) {
				case FormOperation.add:
					DATA.id = 0;
					DATA.fromData.dictTypeId = dictTypeId;
					DATA.fromData.name = '';
					DATA.fromData.value = '';
					DATA.fromData.state = undefined;
					DATA.fromData.flag = undefined;
					DATA.fromData.desc = '';

					break;
				case FormOperation.edit:
					if (!item) return false;
					DATA.id = item.id;
					DATA.fromData.dictTypeId = dictTypeId;
					DATA.fromData.name = item.name;
					DATA.fromData.state = item.state;
					DATA.fromData.value = item.value;
					DATA.fromData.flag = item.flag;
					DATA.fromData.desc = item.desc;
					DATA.fromData.order = item.order;
					//可以在这里写一些use 获取其他的数据
					break;
			}
			return true;
		};

		const rules: FormRules = {
			name: [{ required: true, message: '请输入名称', trigger: 'blur' }],
			value: [{ required: true, message: '请输入编码', trigger: 'blur' }],
			state: [{ required: true, message: '请选择状态', trigger: 'blur' }],
			order: [{ required: true, message: '排序不能为空', trigger: 'blur' }],
		};

		const save = () => {
			//提交数据
			let promist: Promise<api_type.IResponse>;
			switch (DATA.operation) {
				case FormOperation.add:
					promist = svc.add_svc(DATA.fromData);
					break;
				case FormOperation.edit:
					promist = svc.edit_svc(DATA.id, DATA.fromData);
					break;
				default:
					return;
			}
			showLoading();
			promist
				.then((res) => {
					diaglogForm.value?.closeDialog();
					ElMessage.success(`操作成功`);
					ctx.emit('saved', res.data);
				})
				.finally(() => {
					closeLoading();
				});
		};
		const fromSlots = {
			buttons: () => (
				<>
					<ElButton
						onClick={() => {
							diaglogForm.value?.validate(save);
						}}>
						保存
					</ElButton>
				</>
			),
			default: () => (
				<>
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="名称" prop="name">
								<ElInput v-model={DATA.fromData.name} placeholder="名称" />
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="值" prop="value">
								<ElInput v-model={DATA.fromData.value} placeholder="值" />
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="标志" prop="flag">
								<ElInput
									v-model={DATA.fromData.flag}
									placeholder="标志值,可空"></ElInput>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="排序" prop="order">
								<ElInputNumber
									v-model={DATA.fromData.order}
									placeholder="排序"
								/>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElRow>
						<ElCol>
							<ElFormItem label="状态" prop="state">
								<StateSelect
									v-model={DATA.fromData.state}
									placeholder="请选择"
								/>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElRow>
						<ElCol>
							<ElFormItem label="描述" prop="desc">
								<ElInput
									v-model={DATA.fromData.desc}
									type="textarea"
									placeholder="描述"
								/>
							</ElFormItem>
						</ElCol>
					</ElRow>
				</>
			),
		};

		const rander = (): VNode => {
			return (
				<DialogForm
					title={prop.title}
					labelWidth={prop.labelWidth}
					style={ctx.attrs}
					rules={rules}
					ref={diaglogForm}
					v-slots={fromSlots}
				/>
			);
		};
		const openDialog = (
			oper: FormOperation,
			dictTypeId: number,
			item?: Item
		) => {
			init_data(oper, dictTypeId, item);
			diaglogForm.value?.openDialog();
		};
		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
		return rander;
	}, //end setup
});
