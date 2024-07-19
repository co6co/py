import { defineComponent, ref, reactive, provide, computed } from 'vue';
import type { InjectionKey } from 'vue';
import 'md-editor-v3/lib/style.css';
import * as api from '@/api/mp';
import { MdEditor } from 'md-editor-v3';
import { get_store } from '@/hooks/wx';
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

import {
	ElRow,
	ElCol,
	ElButton,
	ElSelect,
	ElFormItem,
	ElInput,
	ElMessage,
	ElOption,
	type FormRules,
} from 'element-plus';

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
		const example = `
		示例代码：
		{
			"button": [
				{
					"name": "一级菜单1",
					"sub_button": [
						{
							"type": "view",
							"name": "跳转到主页",
							"url": "http://www.example.com"
						},
						{
							"type": "click",
							"name": "发送欢迎消息",
							"key": "welcome_message"
						}
					]
				},
				{
					"name": "一级菜单2",
					"sub_button": [
						{
							"type": "view",
							"name": "查看产品列表",
							"url": "http://www.example.com/products"
						},
						{
							"type": "view",
							"name": "联系客服",
							"url": "http://www.example.com/contact"
						}
					]
				},
				{
					"name": "一级菜单3",
					"sub_button": [
						{
							"type": "view",
							"name": "最新资讯",
							"url": "http://www.example.com/news"
						},
						{
							"type": "view",
							"name": "常见问题",
							"url": "http://www.example.com/faq"
						}
					]
				}
			]
		}
		`;
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const config = get_store();
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
							DATA.fromData.content = example;
						}}>
						复制示例
					</ElButton>
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
							<ElFormItem label="公众号" prop="openId">
								<ElSelect
									style="width:60%"
									clearable={true}
									v-model={DATA.fromData.openId}
									placeholder="所属公众号">
									{config.list.map((item, index: number) => {
										return (
											<ElOption
												key={index}
												label={item.name}
												value={item.openId}></ElOption>
										);
									})}
								</ElSelect>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="名称" prop="name">
								<ElInput
									style="width:60%"
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
							placeholder={example}
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
