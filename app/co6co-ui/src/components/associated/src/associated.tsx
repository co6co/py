import {
	defineComponent,
	ref,
	reactive,
	provide,
	computed,
	nextTick,
} from 'vue';
import type { InjectionKey } from 'vue';

import {
	DialogForm,
	type ObjectType,
	type FormData,
	FormOperation,
	type ITreeSelect,
	type IAssociation,
	type IResponse,
	DialogFormInstance,
} from 'co6co';

import {
	ElRow,
	ElCol,
	ElButton,
	ElFormItem,
	ElMessage,
	type FormRules,
	ElTreeV2,
	ElTree,
	ElCard,
	ElEmpty,
	ElCheckbox,
} from 'element-plus';
import {
	type TreeNodeData,
	type TreeKey,
} from 'element-plus/es/components/tree/src/tree.type';

import { minus, traverseTreeData, showLoading, closeLoading } from 'co6co';
import { tree_props } from '@/constants';

//Omit、Pick、Partial、Required
//export type FormItem = Omit<Item, 'createUser' | 'updateUser' | 'createTime' | 'updateTime'>
export type FormItem = IAssociation;

/**
 * 关联模型
 */
export type AssociatedSelect = ITreeSelect & {
	associatedValue: number | boolean;
};
export type get_svc = (
	associatedId: number,
	data: any
) => Promise<IResponse<AssociatedSelect[]>>;
export type save_svc = (
	associatedId: number,
	data: IAssociation
) => Promise<IResponse>;
export type filter = (treeItem: any) => boolean;

interface FormTree {
	treeSelectData: ITreeSelect[];

	/**
	 *
	 * tree 默认选中
	 */
	treeDefaultChecked: TreeKey[];
	/**
	 * 被关联所有值【通常为ID，比如菜单ID，关联id为，角色ID】
	 */
	allIds: number[];
	/**
	 * 是否选中所有按钮
	 */
	allChecked: boolean;
	save_svc?: save_svc;
	filter?: filter;
}

/**
 * 关联
 * 树形结构保存 多对多关系
 */
export default defineComponent({
	name: 'associatedTree', //关联模型，树形结构保存 多对多关系
	props: {
		title: {
			type: String,
		},
		labelWidth: {
			type: Number, //as PropType<ObjectConstructor>,
			default: 90,
		},
		/**
		 * 是否严格的遵循父子不互相关联的做法
		 * default:false
		 */
		checkStrictly: {
			type: Boolean,
			default: false,
		},
	},
	emits: {
		saved: (data: any) => true,
	},
	setup(prop, ctx) {
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<FormData<number, FormItem> & FormTree>({
			treeSelectData: [],
			operation: FormOperation.add,
			id: 0,
			fromData: { add: [], remove: [] },
			allChecked: false,
			allIds: [],
			treeDefaultChecked: [],
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const init_data = (associatedId: number, get_association_svc: get_svc) => {
			DATA.operation = FormOperation.edit;
			switch (DATA.operation) {
				case FormOperation.edit:
					if (!associatedId) return false;
					DATA.treeDefaultChecked = [];
					DATA.allIds = [];

					get_association_svc(associatedId, {}).then((res) => {
						DATA.id = associatedId;
						DATA.treeSelectData = res.data;
						traverseTreeData(res.data, (d) => {
							if (DATA.filter)
								DATA.allIds.push(...[d].filter(DATA.filter).map((m) => m.id));
							else DATA.allIds.push(d.id);

							//选中的值
							const dt = d as AssociatedSelect;
							if (dt.associatedValue) {
								if (DATA.filter)
									DATA.treeDefaultChecked.push(
										...[d].filter(DATA.filter).map((m) => m.id)
									);
								else DATA.treeDefaultChecked.push(d.id);
							}
						});
						//解决因第一项选中，后面加载的后仍被选中问题
						elTreeRef.value?.setCheckedKeys(DATA.treeDefaultChecked);

						nextTick(() => {
							isAllChecked();
						});
						DATA.fromData.add = [];
						DATA.fromData.remove = [];
						console.info('ALLIDS_1', DATA.allIds.length);
					});

					break;
			}
			return true;
		};
		var onValidator = (
			rule: any,
			value: any,
			callback: (error?: string | Error | undefined) => void
		) => {
			if (
				(DATA.fromData.add && DATA.fromData.add.length > 0) ||
				(DATA.fromData.remove && DATA.fromData.remove.length > 0)
			) {
				callback();
			} else {
				callback(new Error('选择未改变,不需要保存！'));
			}
		};
		const rules_b: FormRules = {
			treeData: [
				{ required: true, validator: onValidator, trigger: ['blur', 'change'] },
			],
		};
		const saveBefore = () => {
			let nodes = elTreeRef.value?.getCheckedNodes();
			if (nodes) {
				const ids = nodes.map((m) => m.id);
				DATA.fromData.add = minus(ids, DATA.treeDefaultChecked);
				DATA.fromData.remove = minus(DATA.treeDefaultChecked, ids);
			}
		};

		const save = () => {
			//提交数据
			if (DATA.save_svc) {
				showLoading();
				DATA.save_svc(DATA.id, DATA.fromData)
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
			}
		};
		//@ts-ignore
		const rightCardSlots = {
			header: () => {
				return (
					<ElFormItem
						label="请选择相关功能"
						labelWidth={135}
						prop="treeData"></ElFormItem>
				);
			},
		};
		const elTreeRef = ref<InstanceType<typeof ElTreeV2>>();
		const onAllCheck = () => {
			if (DATA.allChecked) {
				let checked: Array<number> = [];
				traverseTreeData(DATA.treeSelectData, (d) => {
					//@ts-ignore
					const dt = d as AssociatedSelect;
					checked.push(d.id);
				});
				elTreeRef.value?.setCheckedKeys(checked);
			} else {
				elTreeRef.value?.setCheckedKeys([]);
			}
		};
		//是否被全选
		const isAllChecked = () => {
			let nodes = elTreeRef.value?.getCheckedNodes();
			if (nodes) {
				let checkedIds: Array<number> = [];
				if (DATA.filter)
					checkedIds = nodes.filter((m) => DATA.filter!(m)).map((m) => m.id);
				else checkedIds = nodes.map((m) => m.id);
				if (checkedIds.length == DATA.allIds.length) DATA.allChecked = true;
				else DATA.allChecked = false;
			}
		};
		//选择发生改变
		const onCheck = (data: TreeNodeData, checked: boolean) => {
			nextTick(() => {
				isAllChecked();
			});
		};
		const fromSlots = {
			buttons: () => (
				<>
					<ElButton
						onClick={() => {
							diaglogForm.value?.validate(save, saveBefore);
						}}>
						保存
					</ElButton>
				</>
			),
			default: () => (
				<>
					<ElRow>
						<ElCol>
							<ElCard>
								<div style="height: 100%; overflow: auto">
									{DATA.treeSelectData && DATA.treeSelectData.length > 0 ? (
										<>
											{DATA.allChecked}
											<ElCheckbox
												v-model={DATA.allChecked}
												onChange={onAllCheck}
												style="margin-left: 23px;">
												全选
											</ElCheckbox>

											<ElTree
												check-strictly={prop.checkStrictly}
												check-on-click-node={true}
												showCheckbox={true}
												default-expand-all={true}
												nodeKey="id"
												ref={elTreeRef}
												props={tree_props}
												data={DATA.treeSelectData}
												defaultCheckedKeys={DATA.treeDefaultChecked}
												onCheck-change={onCheck}></ElTree>
										</>
									) : (
										<ElEmpty description="未加载数据" />
									)}
								</div>
							</ElCard>
						</ElCol>
					</ElRow>
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
					v-slots={fromSlots}
				/>
			);
		};
		const openDialog = (
			associationId: number,
			get_association_svc: get_svc,
			save_association_svc: save_svc,
			filter?: filter
		) => {
			init_data(associationId, get_association_svc);
			DATA.save_svc = save_association_svc;
			DATA.filter = filter;
			diaglogForm.value?.openDialog();
		};
		const update = () => {};
		ctx.expose({
			openDialog,
			update,
		});
		rander.openDialog = openDialog;
		rander.update = update;
		return rander;
	}, //end setup
});
