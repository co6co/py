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
	DialogFormInstance,
	showLoading,
	closeLoading,
} from 'co6co';
import { type FormData, FormOperation } from 'co6co';
import * as api_type from 'co6co';
import { retsetPwd_svc } from '@/api/sys/user';
import { useTree } from '@/hooks/useUserGroupSelect';
import { useState } from '@/hooks/useUserSelect';
import {
	ElButton,
	ElFormItem,
	ElInput,
	ElMessage,
	type FormRules,
} from 'element-plus';

export interface Item extends api_type.FormItemBase {
	userName?: string;
	password?: string;
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
			default: 120,
		},
	},
	emits: {
		//@ts-ignore
		saved: (data: any) => true,
	},
	setup(prop, ctx) {
		//@ts-ignore
		const hookTree = useTree(0);
		//@ts-ignore
		const { loadData, selectData } = useState();
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<FormData<number, FormItem> & { category: number }>({
			operation: FormOperation.add,
			id: 0,
			fromData: { userName: '' },
			category: 0,
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);
		onMounted(async () => {
			await hookTree.loadData();
			await loadData();
		});
		const init_data = (item?: Item) => {
			DATA.operation = api_type.FormOperation.edit;
			if (!item) return false;

			DATA.fromData.userName = item.userName;
			DATA.fromData.password = '';
			return true;
		};
		const rules_base: FormRules = {
			userName: [
				{ required: true, message: '请输入用户名', trigger: ['blur'] },
			],
			password: [
				{
					required: true,
					min: 6,
					max: 256,
					message: '请输入6-256位字符',
					trigger: ['blur', 'change'],
				},
			],
		};
		const passwordShowTxt = computed(() => {
			return DATA.category == 2 ? '新AccessToken' : '新密码';
		});
		const passwordShowType = computed(() => {
			return DATA.category == 2 ? 'textarea' : 'password';
		});
		const rules = computed(() => rules_base);
		const save = () => {
			//提交数据
			let promist: Promise<api_type.IResponse> = retsetPwd_svc(DATA.fromData);
			showLoading();
			promist
				.then((res) => {
					diaglogForm.value?.closeDialog();
					ElMessage.success(`重置密码成功`);
					update();
					ctx.emit('saved', res.data);
				})
				.catch((e) => {
					if (e.message) ElMessage.error(`重置密码失败:${e.message}`);
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
						<ElInput v-model={DATA.fromData.userName} readonly></ElInput>
					</ElFormItem>

					<ElFormItem label={passwordShowTxt.value} prop="password">
						<ElInput
							v-model={DATA.fromData.password}
							type={passwordShowType.value}
							showPassword={true}
							placeholder={passwordShowTxt.value}
						/>
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
		const openDialog = (category: number, item?: Item) => {
			init_data(item);
			DATA.category = category;
			diaglogForm.value?.openDialog();
		};
		const update = () => {
			hookTree.refresh().then();
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
