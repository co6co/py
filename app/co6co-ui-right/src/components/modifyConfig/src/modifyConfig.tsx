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
	FormOperation,
	showLoading,
	closeLoading,
	type DialogFormInstance,
	type FormData,
} from 'co6co';

import * as api_type from 'co6co';
import { configSvc as svc } from '@/api/config';
import { Flag } from '@/constants';
import { useDictTypeSelect, useDictSelect } from '@/hooks/useDictState';
import { validatorBack } from '@/constants';
import {
	ElRow,
	ElCol,
	ElButton,
	ElFormItem,
	ElInput,
	ElMessage,
	type FormRules,
	ElSelect,
	ElOption,
} from 'element-plus';

export interface Item extends api_type.FormItemBase {
	id: number;
	name: string;
	code: string;
	sysFlag: Flag; //Y|N
	dictFlag: Flag; //Y|N
	dictTypeId?: number;
	value: string;
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
			default: 110,
		},
	},
	emits: {
		//@ts-ignore
		saved: (data: any) => true,
	},
	setup(prop, ctx) {
		const { loadData, selectData } = useDictTypeSelect();

		const diaglogForm = ref<DialogFormInstance>();

		const DATA = reactive<FormData<number, FormItem>>({
			operation: FormOperation.add,

			id: 0,
			fromData: {
				name: '',
				code: '',
				sysFlag: Flag.N,
				dictFlag: Flag.N,
				value: '',
			},
		});
		const editData = reactive<{ sysFlag: Flag; dictFlag: Flag }>({
			sysFlag: Flag.N,
			dictFlag: Flag.N,
		});
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);

		const init_data = (oper: FormOperation, item?: Item) => {
			DATA.operation = oper;
			editData.sysFlag = Flag.N;
			editData.dictFlag = Flag.N;
			switch (oper) {
				case FormOperation.add:
					DATA.id = 0;
					DATA.fromData.name = '';
					DATA.fromData.code = '';
					DATA.fromData.sysFlag = Flag.N;
					DATA.fromData.dictFlag = Flag.N;
					DATA.fromData.value = '';
					DATA.fromData.remark = '';

					break;
				case FormOperation.edit:
					if (!item) return false;
					DATA.id = item.id;
					Object.assign(DATA.fromData, item);
					//可以在这里写一些use 获取其他的数据
					//从后台获取的数据有系统标识，是有些数据就不允许更改
					editData.sysFlag = item.sysFlag;
					editData.dictFlag = item.dictFlag;
					//加载字典
					if (item.dictTypeId) onDictTypeChange();
					break;
			}
			return true;
		};
		const validCode = (rule: any, value: any, callback: validatorBack) => {
			svc.exist_svc(value, DATA.id).then((res) => {
				if (res.data)
					return (
						(rule.message = res.message), callback(new Error(rule.message))
					);
				else return callback();
			});
		};
		const rules: FormRules = {
			name: [{ required: true, message: '请输入名称', trigger: 'blur' }],
			code: [
				{ required: true, message: '请输入编码', trigger: 'blur' },
				{ min: 3, max: 64, message: '编码长度应3-64个字符', trigger: 'blur' },
				{ validator: validCode, trigger: 'blur' },
			],

			sysFlag: [{ required: true, message: '请选择标识', trigger: 'blur' }],
			order: [{ required: true, message: '排序不能为空', trigger: 'blur' }],
		};

		const save = () => {
			//提交数据
			let promist: Promise<api_type.IResponse>;
			switch (DATA.operation) {
				case FormOperation.add:
					promist = svc.add_svc(DATA.fromData);
					break;
				case FormOperation.edit:
					promist = svc.edit_svc(DATA.id, DATA.fromData);
					break;
				default:
					return;
			}
			showLoading();
			promist
				.then((res) => {
					diaglogForm.value?.closeDialog();
					ElMessage.success(`操作成功`);
					ctx.emit('saved', res.data);
				})
				.finally(() => {
					closeLoading();
				});
		};
		const systemFlage = computed(() => {
			return editData.sysFlag == Flag.Y;
		});
		const dictFlage = computed(() => {
			return editData.dictFlag == Flag.Y;
		});
		onMounted(async () => {
			await loadData();
		});
		const dictSelect = useDictSelect();
		const onDictTypeChange = () => {
			if (DATA.fromData.dictTypeId) {
				showLoading();
				dictSelect
					.query(DATA.fromData.dictTypeId)
					.then(() => {})
					.finally(() => {
						closeLoading();
					});
			}
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
								<ElInput
									v-model={DATA.fromData.name}
									placeholder="名称"></ElInput>
							</ElFormItem>
						</ElCol>
						<ElCol span={12}>
							<ElFormItem label="系统配置" prop="state">
								<ElSelect
									disabled={systemFlage.value}
									v-model={DATA.fromData.sysFlag}
									placeholder="系统配置">
									{Object.keys(Flag).map((d, index) => {
										return (
											<ElOption
												key={index}
												label={d == Flag.Y ? '是' : '否'}
												value={d}></ElOption>
										);
									})}
								</ElSelect>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElRow>
						<ElCol>
							<ElFormItem label="配置编码" prop="code">
								<ElInput
									disabled={systemFlage.value}
									v-model={DATA.fromData.code}
									placeholder="配置编码"></ElInput>
							</ElFormItem>
						</ElCol>
					</ElRow>
					<ElRow>
						<ElCol>
							<ElFormItem label="使用字典" prop="dictFlag">
								<ElSelect
									disabled={systemFlage.value && dictFlage.value}
									v-model={DATA.fromData.dictFlag}
									placeholder="使用字典">
									{Object.keys(Flag).map((d, index) => {
										return (
											<ElOption
												key={index}
												label={d == Flag.Y ? '是' : '否'}
												value={d}></ElOption>
										);
									})}
								</ElSelect>
							</ElFormItem>
						</ElCol>
					</ElRow>

					{DATA.fromData.dictFlag == Flag.Y ? (
						<ElRow>
							<ElCol span={12}>
								<ElFormItem label="选择字典" prop="dictTypeId">
									<ElSelect
										disabled={systemFlage.value}
										v-model={DATA.fromData.dictTypeId}
										placeholder="字典"
										onChange={onDictTypeChange}>
										{selectData.value.map((d, index) => {
											return (
												<ElOption
													key={index}
													label={d.name}
													value={d.id}></ElOption>
											);
										})}
									</ElSelect>
								</ElFormItem>
							</ElCol>
							<ElCol span={12}>
								<ElFormItem label="配置值" prop="value">
									<ElSelect v-model={DATA.fromData.value} placeholder="配置值">
										{dictSelect.selectData.value.map((d, index) => {
											return (
												<ElOption
													key={index}
													label={d.name}
													value={d.value}></ElOption>
											);
										})}
									</ElSelect>
								</ElFormItem>
							</ElCol>
						</ElRow>
					) : (
						<ElRow>
							<ElCol>
								<ElFormItem label="配置值" prop="value">
									<ElInput
										type="textarea"
										v-model={DATA.fromData.value}
										placeholder="配置值"></ElInput>
								</ElFormItem>
							</ElCol>
						</ElRow>
					)}
					<ElRow>
						<ElCol>
							<ElFormItem label="备注" prop="desc">
								<ElInput
									v-model={DATA.fromData.remark}
									type="textarea"
									placeholder="备注"></ElInput>
							</ElFormItem>
						</ElCol>
					</ElRow>
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
		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
		return rander;
	}, //end setup
});
