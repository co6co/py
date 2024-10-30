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
import api from '@/api/sys/user';
import { useTree } from '@/hooks/useUserGroupSelect';
import { useState } from '@/hooks/useUserSelect';
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
	userName?: string;
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
		const { treeSelectData, refresh } = useTree(0);
		const { selectData } = useState();
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,
			id: 0,
			fromData: { userName: '', state: 0, userGroupId: 0 },
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const init_data = (oper: FormOperation, item?: Item) => {
			DATA.operation = oper;
			switch (oper) {
				case FormOperation.add:
					DATA.id = 0;
					DATA.fromData.userName = '';
					DATA.fromData.state = 0;
					DATA.fromData.password = '';
					DATA.fromData.userGroupId = undefined;
					break;
				case FormOperation.edit:
					if (!item) return false;
					DATA.id = item.id;
					DATA.fromData.userName = item.userName;
					DATA.fromData.state = item.state;
					DATA.fromData.userGroupId = item.userGroupId;
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
			state: [{ required: true, message: '请用户组编码', trigger: 'blur' }],
		};
		const rules_add: FormRules = {
			...{
				password: [
					{
						required: true,
						min: 6,
						max: 20,
						message: '请输入6-20位密码',
						trigger: 'blur',
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
				.then((res) => {
					diaglogForm.value?.closeDialog();
					ElMessage.success(`操作成功`);
					refresh();
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
					<ElFormItem label="用户名" prop="userName">
						<ElInput
							v-model={DATA.fromData.userName}
							placeholder="用户名"></ElInput>
					</ElFormItem>
					<ElFormItem label="所属用户组" prop="userGroupId">
						<ElTreeSelect
							v-model={DATA.fromData.userGroupId}
							multiple={false}
							check-strictly
							props={{ children: 'children', label: 'name', value: 'id' }}
							data={treeSelectData.value}></ElTreeSelect>
					</ElFormItem>
					{DATA.operation == FormOperation.add ? (
						<ElFormItem label="密码" prop="password">
							<ElInput
								v-model={DATA.fromData.password}
								type="password"
								showPassword={true}
								placeholder="密码"></ElInput>
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
		const update = () => {
			refresh();
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
