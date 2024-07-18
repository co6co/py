import { defineComponent, ref, reactive, computed, provide } from 'vue';
import type { InjectionKey } from 'vue';
import {
	DialogForm,
	FormOperation,
	showLoading,
	closeLoading,
	type DialogFormInstance,
	type ObjectType,
	type FormData,
} from 'co6co';

import * as api_type from 'co6co';
import { dictTypeSvc as svc } from '@/api/dict';
import { Flag } from '@/constants';
import { useState } from '@/hooks/useDictState';
import { validatorBack } from '@/constants';
import {
	ElRow,
	ElCol,
	ElButton,
	ElFormItem,
	ElInput,
	ElMessage,
	type FormRules,
	ElSelect,
	ElOption,
	ElInputNumber,
} from 'element-plus';

export interface Item extends api_type.FormItemBase {
	id: number;
	name?: string;
	code?: string;
	state?: number;
	sysFlag: Flag; //Y|N
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
		const { selectData } = useState();
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,
			id: 0,
			fromData: { sysFlag: Flag.N, order: 0 },
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const init_data = (oper: FormOperation, item?: Item) => {
			DATA.operation = oper;
			switch (oper) {
				case FormOperation.add:
					DATA.id = 0;
					DATA.fromData.name = '';
					DATA.fromData.state = 0;
					DATA.fromData.name = '';
					DATA.fromData.code = '';
					DATA.fromData.sysFlag = Flag.N;
					DATA.fromData.desc = '';
					DATA.fromData.order = 0;

					break;
				case FormOperation.edit:
					if (!item) return false;
					DATA.id = item.id;
					DATA.fromData.name = item.name;
					DATA.fromData.state = item.state;
					DATA.fromData.name = item.name;
					DATA.fromData.code = item.code;
					DATA.fromData.sysFlag = item.sysFlag;
					DATA.fromData.desc = item.desc;
					DATA.fromData.order = item.order;
					//可以在这里写一些use 获取其他的数据
					break;
			}
			return true;
		};
		const validCode = (rule: any, value: any, callback: validatorBack) => {
			svc.exist_svc(value, DATA.id).then((res) => {
				if (res.data)
					return (
						(rule.message = res.message), callback(new Error(rule.message))
					);
				else return callback();
			});
		};
		const rules: FormRules = {
			name: [{ required: true, message: '请输入名称', trigger: 'blur' }],
			code: [
				{ required: true, message: '请输入编码', trigger: 'blur' },
				{ min: 3, max: 10, message: '编码长度应3个字符', trigger: 'blur' },
				{ validator: validCode, trigger: 'blur' },
			],
			state: [{ required: true, message: '请选择状态', trigger: 'blur' }],
			sysFlag: [{ required: true, message: '请选择标识', trigger: 'blur' }],
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
					if (res.code == 0) {
						diaglogForm.value?.closeDialog();
						ElMessage.success(`操作成功`);
						ctx.emit('saved', res.data);
					} else {
						ElMessage.error(`操作失败:${res.message}`);
					}
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
								<ElInput
									v-model={DATA.fromData.name}
									placeholder="名称"></ElInput>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="系统字典" prop="state">
								<ElSelect v-model={DATA.fromData.sysFlag} placeholder="请选择">
									{Object.keys(Flag).map((d, index) => {
										return (
											<ElOption
												key={index}
												label={d == Flag.Y ? '是' : '否'}
												value={d}></ElOption>
										);
									})}
								</ElSelect>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElRow>
						<ElCol>
							<ElFormItem label="编码" prop="code">
								<ElInput
									v-model={DATA.fromData.code}
									placeholder="编码"></ElInput>
							</ElFormItem>
						</ElCol>
					</ElRow>

					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="状态" prop="state">
								<ElSelect v-model={DATA.fromData.state} placeholder="请选择">
									{selectData.value.map((d, _) => {
										return (
											<ElOption
												key={d.key}
												label={d.label}
												value={d.value}></ElOption>
										);
									})}
								</ElSelect>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="排序" prop="talkbackNo">
								<ElInputNumber
									v-model={DATA.fromData.order}
									placeholder="排序"></ElInputNumber>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElRow>
						<ElCol>
							<ElFormItem label="描述" prop="desc">
								<ElInput
									v-model={DATA.fromData.desc}
									type="textarea"
									placeholder="描述"></ElInput>
							</ElFormItem>
						</ElCol>
					</ElRow>
				</>
			),
		};

		const rander = (): ObjectType => {
			return (
				<DialogForm
					title={prop.title}
					labelWidth={prop.labelWidth}
					style={ctx.attrs}
					rules={rules.value}
					ref={diaglogForm}
					v-slots={fromSlots}
				/>
			);
		};
		const openDialog = (oper: FormOperation, item?: Item) => {
			init_data(oper, item);
			diaglogForm.value?.openDialog();
		};
		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
		return rander;
	}, //end setup
});
