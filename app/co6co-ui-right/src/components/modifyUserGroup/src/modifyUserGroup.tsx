import { defineComponent, ref, reactive, provide, VNode, onMounted } from 'vue';
import type { InjectionKey } from 'vue';
import {
	DialogForm,
	showLoading,
	closeLoading,
	type FormData,
	FormOperation,
} from 'co6co';

import * as api_type from 'co6co';
import api from '@/api/sys/userGroup';
import { useTree } from '@/hooks/useUserGroupSelect';
import {
	ElButton,
	ElFormItem,
	ElInputNumber,
	ElInput,
	ElMessage,
	type FormRules,
	ElTreeSelect,
} from 'element-plus';

export interface Item extends api_type.FormItemBase {
	id: number;
	parentId?: number;
	name?: string;
	code?: string;
	order?: number;
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
		const { loadData, treeSelectData, refresh } = useTree();
		const diaglogForm = ref<InstanceType<typeof DialogForm>>();
		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,
			id: 0,
			fromData: {},
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);
		onMounted(async () => {
			await loadData();
		});
		const init_data = (oper: FormOperation, item?: Item) => {
			DATA.operation = oper;
			switch (oper) {
				case FormOperation.add:
					DATA.id = -1;
					DATA.fromData.name = '';
					DATA.fromData.code = '';
					DATA.fromData.parentId = 0;

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
		const rules: FormRules = {
			name: [{ required: true, message: '请输入组名称', trigger: 'blur' }],
			parentId: [{ required: true, message: '请选择父节点', trigger: 'blur' }],
			code: [{ required: true, message: '请用户组编码', trigger: 'blur' }],
			order: [{ required: true, message: '请用排序编码', trigger: 'blur' }],
		};

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
					<ElFormItem label="名称" prop="name">
						<ElInput
							v-model={DATA.fromData.name}
							placeholder="用户组名称"></ElInput>
					</ElFormItem>
					<ElFormItem label="父节点" prop="parentId">
						<ElTreeSelect
							v-model={DATA.fromData.parentId}
							multiple={false}
							check-strictly
							props={{ children: 'children', label: 'name', value: 'id' }}
							data={treeSelectData.value}></ElTreeSelect>
					</ElFormItem>
					<ElFormItem label="代码" prop="code">
						<ElInput
							v-model={DATA.fromData.code}
							placeholder="用户组代码"></ElInput>
					</ElFormItem>
					<ElFormItem label="排序" prop="order">
						<ElInputNumber
							v-model={DATA.fromData.order}
							placeholder="排序"></ElInputNumber>
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
					rules={rules}
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
