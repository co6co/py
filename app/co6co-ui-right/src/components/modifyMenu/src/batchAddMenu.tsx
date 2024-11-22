import { defineComponent, ref, reactive, provide, VNode } from 'vue';
import type { InjectionKey } from 'vue';
import {
	DialogForm,
	showLoading,
	closeLoading,
	DialogFormInstance,
	type FormData,
	FormOperation,
	type IResponse,
} from 'co6co';

import { batch_add_svc } from '@/api/sys/menu';
import { useTree, useMenuState, MenuCateCategory } from '@/hooks/useMenuSelect';
import { usePageFeature } from '@/hooks/useMethods';
import {
	ElRow,
	ElCol,
	ElButton,
	ElFormItem,
	ElInput,
	ElMessage,
	type FormRules,
	ElTreeSelect,
	ElSelectV2,
	ElInputNumber,
} from 'element-plus';

interface Item {
	id?: number;
	parentId?: number;
	name?: string;
	code?: string;
	category: MenuCateCategory;
	permissionKey?: string;
	methods?: string[] | string;
	order?: number;
	status?: number;
	remark?: string;
}

export default defineComponent({
	name: 'addSubMenus',
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
		saved: (data: any) => true,
	},
	setup(prop, ctx) {
		const { treeSelectData, refresh } = useTree();
		const menuStateData = useMenuState();
		const pageFeature = usePageFeature();
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<
			FormData<number, Array<Item>> & {
				base: Item & { methods?: Array<string> };
			}
		>({
			operation: FormOperation.add,
			id: 0,
			base: { category: MenuCateCategory.Button },
			fromData: [{ category: MenuCateCategory.Button }],
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.base);

		const init_data = (item: Item) => {
			DATA.id = -1;
			DATA.base.name = item.name;
			DATA.base.code = item.code;
			DATA.base.parentId = item.id;
			DATA.base.category = MenuCateCategory.Button;
			if (
				menuStateData.selectData.value &&
				menuStateData.selectData.value.length > 0
			)
				DATA.base.status = menuStateData.selectData.value[0].value as number;
			else DATA.base.status = undefined;

			return true;
		};
		const rules: FormRules = {
			name: [{ required: true, message: '请输入菜单称', trigger: 'blur' }],
			parentId: [{ required: true, message: '请选择父节点', trigger: 'blur' }],
			code: [{ required: true, message: '请菜单编码', trigger: 'blur' }],
			status: [
				{ required: true, message: '请选择状态', trigger: ['blur', 'change'] },
			],
			methods: [
				{
					required: true,
					message: '请选择方法名',
					trigger: ['blur', 'change'],
				},
			],
		};

		const genPostData = () => {
			const data = DATA.base.methods?.map((method, index) => {
				const item = {
					parentId: DATA.base.parentId,
					name: DATA.base.name + '_' + method,
					code: DATA.base.code + '_' + method,
					category: DATA.base.category,
					permissionKey: method,
					methods: method,
					order: (DATA.base.order ? DATA.base.order : 0) + index,
					status: DATA.base.status,
					remark: DATA.base.remark,
				};
				return item;
			});
			if (data) DATA.fromData = data;
		};
		const save = () => {
			//提交数据
			let promist: Promise<IResponse>;
			genPostData();
			if (!DATA.fromData || DATA.fromData.length == 0) return;
			promist = batch_add_svc(DATA.fromData);
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
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="名称" prop="name">
								<ElInput v-model={DATA.base.name} placeholder="名称" />
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="代码" prop="code">
								<ElInput v-model={DATA.base.code} placeholder="代码" />
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElFormItem label="父节点" prop="parentId">
						<ElTreeSelect
							disabled
							v-model={DATA.base.parentId}
							multiple={false}
							check-strictly
							props={{ children: 'children', label: 'name', value: 'id' }}
							data={treeSelectData.value}
						/>
					</ElFormItem>
					<ElFormItem label="操作类型" prop="methods">
						<ElSelectV2
							multiple
							clearable
							options={pageFeature.selectData.value}
							v-model={DATA.base.methods}
							placeholder="操作类型"
						/>
					</ElFormItem>
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="排序" prop="order">
								<ElInputNumber v-model={DATA.base.order} placeholder="排序" />
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="状态" prop="status">
								<ElSelectV2
									multiple={false}
									options={menuStateData.selectData.value}
									v-model={DATA.base.status}
									placeholder="状态"
								/>
							</ElFormItem>
						</ElCol>
					</ElRow>

					<ElFormItem label="备注" prop="remark">
						<ElInput
							type="textarea"
							clearable={true}
							v-model={DATA.base.remark}
							placeholder="备注"
						/>
					</ElFormItem>
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
					v-slots={fromSlots}></DialogForm>
			);
		};
		const openDialog = (item: Item) => {
			refresh();
			init_data(item);
			diaglogForm.value?.openDialog();
		};

		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
		return rander;
	}, //end setup
});
