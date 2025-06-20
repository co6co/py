import {
	defineComponent,
	ref,
	reactive,
	//provide,
	VNode,
	onMounted,
	computed,
} from 'vue';
//import type { InjectionKey } from 'vue';
import {
	DialogForm,
	showLoading,
	closeLoading,
	DialogFormInstance,
	type FormData,
	FormOperation,
	type IResponse,
} from 'co6co';

import { batch_add_svc, get_one_svc, type IMenuOne } from '@/api/sys/menu';
import { useTree, useMenuState, MenuCateCategory } from '@/hooks/useMenuSelect';
import { queryViewFeature } from '@/hooks/useView';
import { useFeatureSelect } from '@/hooks/useMethods';
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
type FormItem = Partial<IMenuOne> & {
	remark?: string;
	order?: number;
	methods?: string[];
};

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
		const { loadData, treeSelectData, refresh } = useTree();
		const menuStateData = useMenuState();
		const diaglogForm = ref<DialogFormInstance>();
		const FeaturesRef = ref();
		const DATA = reactive<
			FormData<number, FormItem> & {
				postData: Array<Item>;
			}
		>({
			operation: FormOperation.add,
			id: 0,
			fromData: { category: MenuCateCategory.Button, status: 111 },
			postData: [],
		});
		//@ts-ignore
		//const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		//provide('formData', DATA.fromData);
		onMounted(async () => {
			await menuStateData.loadData();
			await loadData();
		});
		const init_data = (item: Item) => {
			DATA.id = -1;
			if (!item.id) {
				ElMessage.error('请选择对应试图在继续!');
				return false;
			}
			get_one_svc(item.id).then((res) => {
				//DATA.fromData = res.data; //不能使用这种方式直接赋值
				Object.assign(DATA.fromData, res.data);
				if (DATA.fromData.component) {
					queryViewFeature(DATA.fromData.component!).then((features) => {
						FeaturesRef.value = features;
					});
				}
			});

			if (
				menuStateData.selectData.value &&
				menuStateData.selectData.value.length > 0
			)
				DATA.fromData.status = menuStateData.selectData.value[0]
					.value as number;
			else DATA.fromData.status = undefined;
			return true;
		};

		const featureSelect = computed(() => {
			return useFeatureSelect(FeaturesRef.value);
		});
		const rules: FormRules = {
			name: [{ required: true, message: '请输入菜单称', trigger: 'blur' }],
			id: [{ required: true, message: '请选择父节点', trigger: 'blur' }],
			code: [{ required: true, message: '请输入代码', trigger: 'blur' }],
			order: [{ required: true, message: '排序不能为空', trigger: 'blur' }],
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
			const data = DATA.fromData.methods?.map((method, index) => {
				const item = {
					parentId: DATA.fromData.id,
					name: DATA.fromData.name + '_' + method,
					code: DATA.fromData.code + '_' + method,
					category: MenuCateCategory.Button,
					permissionKey: method,
					methods: method,
					order: (DATA.fromData.order ? DATA.fromData.order : 0) + index,
					status: DATA.fromData.status,
					remark: DATA.fromData.remark,
				};
				return item;
			});
			if (data) DATA.postData = data;
		};
		const save = () => {
			//提交数据
			let promist: Promise<IResponse>;
			genPostData();
			if (!DATA.postData || DATA.postData.length == 0) return;
			promist = batch_add_svc(DATA.postData);
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
								<ElInput v-model={DATA.fromData.name} placeholder="名称" />
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="代码" prop="code">
								<ElInput v-model={DATA.fromData.code} placeholder="代码" />
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElFormItem label="父节点" prop="id">
						<ElTreeSelect
							disabled
							v-model={DATA.fromData.id}
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
							options={featureSelect.value.selectData.value}
							v-model={DATA.fromData.methods}
							placeholder="操作类型"
						/>
					</ElFormItem>
					<ElRow>
						<ElCol span={12}>
							<ElFormItem label="排序" prop="order">
								<ElInputNumber
									v-model={DATA.fromData.order}
									placeholder="排序"
								/>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="状态" prop="status">
								<ElSelectV2
									multiple={false}
									options={menuStateData.selectData.value}
									v-model={DATA.fromData.status}
									placeholder="状态"
								/>
							</ElFormItem>
						</ElCol>
					</ElRow>

					<ElFormItem label="备注" prop="remark">
						<ElInput
							type="textarea"
							clearable={true}
							v-model={DATA.fromData.remark}
							placeholder="备注"
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
					rules={rules}
					model={DATA.fromData}
					ref={diaglogForm}
					v-slots={fromSlots}
				/>
			);
		};
		const openDialog = (item: Item) => {
			refresh();
			const result = init_data(item);
			if (result) diaglogForm.value?.openDialog();
		};

		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
		return rander;
	}, //end setup
});
