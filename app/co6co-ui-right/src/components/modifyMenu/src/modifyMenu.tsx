import {
	defineComponent,
	ref,
	reactive,
	provide,
	computed,
	VNode,
	onMounted,
} from 'vue';
import type { InjectionKey } from 'vue';
import {
	DialogForm,
	IconSelect,
	showLoading,
	closeLoading,
	DialogFormInstance,
	ViewSelect,
	type FormData,
	FormOperation,
	type IResponse,
	type FormItemBase,
} from 'co6co';

import api from '@/api/sys/menu';
import {
	useTree,
	useMenuCategory,
	useMenuState,
	MenuCateCategory,
} from '@/hooks/useMenuSelect';
import useHttpMethods, { usePageFeature } from '@/hooks/useMethods';
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

export interface Item extends FormItemBase {
	id: number;
	parentId?: number;
	name?: string;
	code?: string;
	category: MenuCateCategory;
	icon?: string;
	url?: string;
	component?: string;
	methods?: string[] | string;
	permissionKey?: string;
	order?: number;
	status?: number;
	remark?: string;
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
		saved: (data: any) => true,
	},
	setup(prop, ctx) {
		const { loadData, treeSelectData, refresh } = useTree();
		const menuStateData = useMenuState();
		const menuCategoryData = useMenuCategory();
		const httpMethods = useHttpMethods();
		const pageFeature = usePageFeature();
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,
			id: 0,
			fromData: { category: MenuCateCategory.GROUP },
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const init_data = (oper: FormOperation, item?: Item) => {
			DATA.operation = oper;
			switch (oper) {
				case FormOperation.add:
					DATA.id = -1;
					DATA.fromData.name = '';
					DATA.fromData.code = '';
					DATA.fromData.parentId = item ? item.id : 0;
					DATA.fromData.category = 0;
					DATA.fromData.icon = undefined;
					DATA.fromData.url = undefined;
					DATA.fromData.methods = undefined;
					DATA.fromData.permissionKey = undefined;
					DATA.fromData.order = 1;
					if (
						menuStateData.selectData.value &&
						menuStateData.selectData.value.length > 0
					)
						DATA.fromData.status = menuStateData.selectData.value[0]
							.value as number;
					else DATA.fromData.status = undefined;
					DATA.fromData.remark = undefined;
					DATA.fromData.component = undefined;
					break;
				case FormOperation.edit:
					if (!item) return false;
					DATA.id = item.id;
					Object.assign(DATA.fromData, item);

					if (DATA.fromData.category == MenuCateCategory.API) {
						DATA.fromData.methods =
							item.methods && (item.methods as string)
								? (item.methods as string).split(',')
								: [];
					} else {
						//button 需要
						DATA.fromData.methods = item.methods;
					}
					//可以在这里写一些use 获取其他的数据
					break;
			}
			return true;
		};
		onMounted(async () => {
			await loadData();
			await menuStateData.loadData();
			await menuCategoryData.loadData();
		});
		const triggers = ['blur', 'change'];
		const trigger = 'blur';
		const rules_b: FormRules = {
			name: [{ required: true, message: '请输入菜单称', trigger: trigger }],
			parentId: [{ required: true, message: '请选择父节点', trigger: trigger }],
			code: [{ required: true, message: '请菜单编码', trigger: trigger }],
			status: [{ required: true, message: '请选择状态', trigger: triggers }],
		};
		const rules_group: FormRules = {
			...{
				permissionKey: [
					{ required: true, message: '权限字', trigger: trigger },
				],
			},
			...rules_b,
		};
		const rules_api: FormRules = {
			...{
				methods: [
					{ required: true, message: '请选择方法名', trigger: trigger },
				],
				url: [{ required: true, message: '请输入链接地址', trigger: trigger }],
			},
			...rules_b,
		};
		const rules_view: FormRules = {
			...{
				icon: [
					{
						required: true,
						message: '请选择视图ICON',
						trigger: triggers,
					},
				],
				url: [{ required: true, message: '请输入视图地址', trigger: trigger }],
				component: [
					{ required: true, message: '请输入视图地址', trigger: triggers },
				],
				permissionKey: [
					{ required: true, message: '权限字', trigger: trigger },
				],
			},
			...rules_b,
		};
		const rules_subView: FormRules = {
			...{
				methods: [
					{
						required: true,
						message: '请选择方法名',
						trigger: triggers,
					},
				],
			},
			...rules_view,
		};
		const rules_button: FormRules = {
			...{
				methods: [
					{
						required: true,
						message: '请选择方法名',
						trigger: triggers,
					},
				],
				permissionKey: [
					{ required: true, message: '权限字', trigger: triggers },
				],
			},
			...rules_b,
		};

		const onFeatueMethod = () => {
			if (!DATA.fromData.permissionKey)
				DATA.fromData.permissionKey = DATA.fromData.methods as string;
		};
		const save = () => {
			//提交数据
			let promist: Promise<IResponse>;
			switch (DATA.operation) {
				case FormOperation.add:
					promist = api.add_svc(DATA.fromData);
					break;
				case FormOperation.edit:
					//提交前是否应该删除其他类别的数据
					promist = api.edit_svc(DATA.id, DATA.fromData);
					break;
				default:
					//没有相关操作
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
					<ElRow>
						<ElCol span={12}></ElCol>
						<ElCol span={12}></ElCol>
					</ElRow>
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="名称" prop="name">
								<ElInput
									v-model={DATA.fromData.name}
									placeholder="菜单名称"></ElInput>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="菜单类别" prop="category">
								<ElSelectV2
									options={menuCategoryData.selectData.value}
									v-model={DATA.fromData.category}
									placeholder="菜单类别"></ElSelectV2>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElFormItem label="父节点" prop="parentId">
						<ElTreeSelect
							v-model={DATA.fromData.parentId}
							multiple={false}
							check-strictly
							props={{ children: 'children', label: 'name', value: 'id' }}
							data={treeSelectData.value}></ElTreeSelect>
					</ElFormItem>
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="代码" prop="code">
								<ElInput
									v-model={DATA.fromData.code}
									placeholder="菜单代码"></ElInput>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="排序" prop="order">
								<ElInputNumber
									v-model={DATA.fromData.order}
									placeholder="排序"></ElInputNumber>
							</ElFormItem>
						</ElCol>
					</ElRow>

					{DATA.fromData.category == MenuCateCategory.API ? (
						<>
							<ElRow>
								<ElCol span={12}>
									<ElFormItem label="URL" prop="url">
										<ElInput
											clearable={true}
											v-model={DATA.fromData.url}
											placeholder="权限字"></ElInput>
									</ElFormItem>
								</ElCol>
								<ElCol span={12}>
									<ElFormItem label="操作类型" prop="methods">
										<ElSelectV2
											multiple={true}
											options={httpMethods.selectData.value}
											v-model={DATA.fromData.methods}
											placeholder="操作类型"></ElSelectV2>
									</ElFormItem>
								</ElCol>
							</ElRow>
						</>
					) : DATA.fromData.category == MenuCateCategory.VIEW ||
					  DATA.fromData.category == MenuCateCategory.SubVIEW ? (
						<>
							<ElRow>
								<ElCol span={12}>
									<ElFormItem label="URL" prop="url">
										<ElInput
											clearable={true}
											v-model={DATA.fromData.url}
											placeholder="视图路径"></ElInput>
									</ElFormItem>
								</ElCol>
								<ElCol span={12}>
									<ElFormItem label="图标" prop="icon">
										<IconSelect v-model={DATA.fromData.icon} />
									</ElFormItem>
								</ElCol>
							</ElRow>
							{DATA.fromData.category == MenuCateCategory.VIEW ? (
								<></>
							) : (
								<ElRow>
									<ElCol>
										<ElFormItem label="操作类型" prop="methods">
											<ElSelectV2
												options={pageFeature.selectData.value}
												v-model={DATA.fromData.methods}
												placeholder="子视图操作类型一般用view,做了特殊出来可选其他"
												onChange={onFeatueMethod}></ElSelectV2>
										</ElFormItem>
									</ElCol>
								</ElRow>
							)}
							<ElRow>
								<ElCol span={12}>
									<ElFormItem label="组件地址" prop="component">
										<ViewSelect
											clearable={true}
											v-model={DATA.fromData.component}
										/>
									</ElFormItem>
								</ElCol>

								<ElCol span={12}>
									<ElFormItem label="权限字" prop="permissionKey">
										<ElInput
											clearable={true}
											v-model={DATA.fromData.permissionKey}
											placeholder="权限字"></ElInput>
									</ElFormItem>
								</ElCol>
							</ElRow>
						</>
					) : DATA.fromData.category == MenuCateCategory.GROUP ? (
						<>
							<ElRow>
								<ElCol span={12}>
									<ElFormItem label="图标" prop="icon">
										<IconSelect v-model={DATA.fromData.icon} />
									</ElFormItem>
								</ElCol>
								<ElCol span={12}>
									<ElFormItem label="权限字" prop="permissionKey">
										<ElInput
											clearable={true}
											v-model={DATA.fromData.permissionKey}
											placeholder="权限字"></ElInput>
									</ElFormItem>
								</ElCol>
							</ElRow>
						</>
					) : (
						<>
							<ElRow>
								<ElCol span={12}>
									<ElFormItem label="操作类型" prop="methods">
										<ElSelectV2
											options={pageFeature.selectData.value}
											v-model={DATA.fromData.methods}
											placeholder="操作类型"
											onChange={onFeatueMethod}></ElSelectV2>
									</ElFormItem>
								</ElCol>
								<ElCol span={12}>
									<ElFormItem label="权限字" prop="permissionKey">
										<ElInput
											clearable={true}
											v-model={DATA.fromData.permissionKey}
											placeholder="权限字"></ElInput>
									</ElFormItem>
								</ElCol>
							</ElRow>
						</>
					)}
					<ElFormItem label="状态" prop="status">
						<ElSelectV2
							multiple={false}
							options={menuStateData.selectData.value}
							v-model={DATA.fromData.status}
							placeholder="状态"
						/>
					</ElFormItem>
					<ElFormItem label="备注" prop="remark">
						<ElInput
							type="textarea"
							clearable={true}
							v-model={DATA.fromData.remark}
							placeholder="备注"></ElInput>
					</ElFormItem>
				</>
			),
		};

		const rules = computed(() => {
			switch (DATA.fromData.category) {
				case MenuCateCategory.API:
					return rules_api;
				case MenuCateCategory.VIEW:
					return rules_view;
				case MenuCateCategory.SubVIEW:
					return rules_subView;
				case MenuCateCategory.Button:
					return rules_button;
				default:
					return rules_group;
			}
		});
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
					v-slots={fromSlots}></DialogForm>
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
