import { defineComponent, ref, reactive, provide, computed } from 'vue';
import type { InjectionKey } from 'vue';

import {
	showLoading,
	closeLoading,
	type FormItemBase,
	type ObjectType,
	type FormData,
	type IResponse,
	FormOperation,
	DialogForm,
} from 'co6co';

import * as api from '@/api/mp';

import {
	ElRow,
	ElCol,
	ElButton,
	ElSelect,
	ElFormItem,
	ElInput,
	ElMessage,
	type FormRules,
} from 'element-plus';
import { MdEditor } from 'md-editor-v3';

import { wx_config_store } from '@/hooks/wx';
import { piniaInstance } from 'co6co';

export interface Item extends FormItemBase {
	id: number;
	openId?: string;
	name?: string;
	content?: string;
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
			default: 90,
		},
	},
	emits: {
		//@ts-ignore
		saved: (data: any) => true,
	},
	setup(prop, ctx) {
		const diaglogForm = ref<InstanceType<typeof DialogForm>>();
		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,
			id: 0,
			fromData: {},
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const config = wx_config_store(piniaInstance);
		const init_data = (oper: FormOperation, item?: Item) => {
			DATA.operation = oper;
			switch (oper) {
				case FormOperation.add:
					DATA.id = -1;
					DATA.fromData.name = '';
					DATA.fromData.openId = '';
					DATA.fromData.content = '';
					break;
				case FormOperation.edit:
					if (!item) return false;
					DATA.id = item.id;
					DATA.fromData.name = item.name;
					DATA.fromData.openId = item.openId;
					DATA.fromData.content = item.content;
					//可以在这里写一些use 获取其他的数据
					break;
			}
			return true;
		};
		const rules_b: FormRules = {
			name: [{ required: true, message: '请输入菜单名', trigger: 'blur' }],
			openId: [{ required: true, message: '请选择公众号', trigger: 'blur' }],
			content: [
				{ required: true, message: '请输入菜单内容', trigger: 'blur,change' },
			],
		};
		const save = () => {
			//提交数据
			let promist: Promise<IResponse>;
			switch (DATA.operation) {
				case FormOperation.add:
					promist = api.add_menu_svc(DATA.fromData);
					break;
				case FormOperation.edit:
					//提交前是否应该删除其他类别的数据
					promist = api.edit_menu_svc(DATA.id, DATA.fromData);
					break;
				default:
					//没有相关操作
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
						<ElCol span={12}></ElCol>
						<ElCol span={12}></ElCol>
					</ElRow>
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="请选择公众号" prop="openId">
								<ElSelect
									options={config.list}
									v-model={DATA.fromData.openId}
									placeholder="菜单类别"
								/>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="菜单名称" prop="name">
								<ElInput
									v-model={DATA.fromData.name}
									placeholder="菜单名称"></ElInput>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElFormItem label="菜单内容" prop="remark">
						<MdEditor
							preview={false}
							class="mgb20"
							v-model={DATA.fromData.content}
						/>
					</ElFormItem>
				</>
			),
		};

		const rules = computed(() => {
			return rules_b;
		});
		const rander = (): ObjectType => {
			return (
				<DialogForm
					title={prop.title}
					labelWidth={prop.labelWidth}
					style={ctx.attrs}
					rules={rules.value}
					ref={diaglogForm}
					v-slots={fromSlots}></DialogForm>
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
