import {
	defineComponent,
	ref,
	reactive,
	computed,
	provide,
	VNode,
	onMounted,
} from 'vue';
import type { InjectionKey } from 'vue';
import {
	DialogForm,
	FormOperation,
	showLoading,
	closeLoading,
	type DialogFormInstance,
	type FormData,
	EnumSelect,
} from 'co6co';

import * as api_type from 'co6co';
import api from '@/api/sys/user';
import { useTree } from '@/hooks/useUserGroupSelect';
import { useState, useCategory } from '@/hooks/useUserSelect';
import {
	ElButton,
	ElFormItem,
	ElInput,
	ElMessage,
	type FormRules,
	ElSelect,
	ElOption,
	ElTreeSelect,
} from 'element-plus';

export interface Item extends api_type.FormItemBase {
	id: number;
	userName: string;
	category: number;
	/**
	 * 用户密码增加用户需要
	 */
	password?: string;
	state: number;
	userGroupId?: number;
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
		const hookTree = useTree(0);
		const { loadData, selectData } = useState();
		const useCategoryHook = useCategory();

		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,
			id: 0,
			fromData: { userName: '', category: 0, state: 0, userGroupId: 0 },
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);
		onMounted(async () => {
			await hookTree.loadData();
			await loadData();
			await useCategoryHook.loadData();
		});
		const init_data = (oper: FormOperation, item?: Item) => {
			DATA.operation = oper;
			switch (oper) {
				case FormOperation.add:
					DATA.id = 0;

					DATA.fromData.userName = '';
					DATA.fromData.state = 0;
					DATA.fromData.category = 0;
					DATA.fromData.password = '';
					DATA.fromData.userGroupId = undefined;
					break;
				case FormOperation.edit:
					if (!item) return false;
					DATA.id = item.id;
					Object.assign(DATA.fromData, item);
					//可以在这里写一些use 获取其他的数据
					break;
			}
			return true;
		};
		const validUserName = (
			rule: any,
			value: any,
			callback: (error?: string | Error | undefined) => void
		) => {
			if (value.length < 3)
				return (
					(rule.message = '长度应该大于3'), callback(new Error(rule.message))
				);
			api.exist_svc(value, DATA.id).then((res) => {
				if (res.data)
					return (
						(rule.message = res.message), callback(new Error(rule.message))
					);
				else return callback();
			});
		};
		const passwordShowTxt = computed(() => {
			return DATA.fromData.category == 2 ? 'AccessToken' : '密码';
		});
		const passwordShowType = computed(() => {
			return DATA.fromData.category == 2 ? 'textarea' : 'password';
		});
		const rules_edit: FormRules = {
			userName: [
				{
					required: true,
					message: '请输入用户名',
					validator: validUserName,
					trigger: 'blur',
				},
			],
			userGroupId: [
				{ required: true, message: '请选择用户组', trigger: 'blur' },
			],
			category: [{ required: true, message: '用户类型', trigger: 'blur' }],
			state: [{ required: true, message: '请用户组编码', trigger: 'blur' }],
		};
		const rules_add: FormRules = {
			...{
				password: [
					{
						required: true,
						min: 6,
						max: 256,
						message: '请输入6-256位字符',
						trigger: ['blur', 'change'],
					},
				],
			},
			...rules_edit,
		};
		const rules = computed(() => {
			switch (DATA.operation) {
				case FormOperation.add:
					return rules_add;
				case FormOperation.edit:
					return rules_edit;
			}
		});

		const save = () => {
			//提交数据
			let promist: Promise<api_type.IResponse>;
			switch (DATA.operation) {
				case FormOperation.add:
					promist = api.add_svc(DATA.fromData);
					break;
				case FormOperation.edit:
					promist = api.edit_svc(DATA.id, DATA.fromData);
					break;
				default:
					return;
			}
			showLoading();
			promist
				.then(async (res) => {
					diaglogForm.value?.closeDialog();
					ElMessage.success(`操作成功`);
					await hookTree.refresh();
					ctx.emit('saved', res.data);
				})
				.finally(() => {
					closeLoading();
				});
		};
		const fromSlots = {
			buttons: () => (
				<ElButton
					onClick={() => {
						diaglogForm.value?.validate(save);
					}}>
					保存
				</ElButton>
			),
			default: () => (
				<>
					<ElFormItem label="用户名" prop="userName">
						<ElInput v-model={DATA.fromData.userName} placeholder="用户名" />
					</ElFormItem>
					<ElFormItem label="用户类别" prop="category">
						<EnumSelect
							clearable
							data={useCategoryHook.selectData.value}
							v-model={DATA.fromData.category}
							style="width:100%"
							placeholder="用户类型"
						/>
					</ElFormItem>
					<ElFormItem label="所属用户组" prop="userGroupId">
						<ElTreeSelect
							v-model={DATA.fromData.userGroupId}
							multiple={false}
							check-strictly
							props={{ children: 'children', label: 'name', value: 'id' }}
							data={hookTree.treeSelectData.value}
						/>
					</ElFormItem>
					{DATA.operation == FormOperation.add ? (
						<ElFormItem label={passwordShowTxt.value} prop="password">
							<ElInput
								v-model={DATA.fromData.password}
								type={passwordShowType.value}
								showPassword={true}
								placeholder={passwordShowTxt.value}
							/>
						</ElFormItem>
					) : (
						<></>
					)}
					<ElFormItem label="状态" prop="state">
						<ElSelect v-model={DATA.fromData.state} placeholder="请选择">
							{selectData.value.map((d, index) => {
								return <ElOption key={index} label={d.label} value={d.value} />;
							})}
						</ElSelect>
					</ElFormItem>
				</>
			),
		};

		const rander = (): VNode => {
			return (
				<DialogForm
					closeOnClickModal={false}
					draggable
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
		const update = () => {
			hookTree.refresh().then(() => {});
		};
		ctx.expose({
			openDialog,
			update,
		});
		rander.openDialog = openDialog;
		rander.update = update;
		return rander;
	}, //end setup
});
