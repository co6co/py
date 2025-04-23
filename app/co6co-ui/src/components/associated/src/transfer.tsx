import {
	defineComponent,
	ref,
	reactive,
	provide,
	onMounted,
	onBeforeUnmount,
	VNode,
	PropType,
	useSlots,
} from 'vue';
import type { InjectionKey } from 'vue';

import {
	ElButton,
	ElMessage,
	ElTransfer,
	type FormRules,
	vLoading,
	ElEmpty,
	ElTooltip,
	ElFormItem,
} from 'element-plus';
import { closeLoading, showLoading } from '@/components/logining';
import { minus } from '@/utils';
import {
	FormData,
	FormOperation,
	IResponse,
	IAssociation,
	ITransferItem,
} from '@/constants';
import DialogForm, { DialogFormInstance } from '@/components/dialogForm';
import { getAssociatedSvc, saveAssociatedSvc } from './associated';
export interface IShowFormat {
	noChecked: string; //'${total}'
	hasChecked: string; //  '${checked}/${total}'
}
interface Option {
	key: number | string;
	label: string;
	disabled: boolean;
	data: ITransferItem;
}
export type createOption = (data: ITransferItem) => Option;
//Omit、Pick、Partial、Required
export default defineComponent({
	name: 'associatedByTransfer',
	props: {
		title: {
			type: String,
		},
		filterable: {
			type: Boolean,
			default: true,
		},
		afterShowTip: {
			type: Number,
			default: 1500,
		},
		format: {
			type: Object as PropType<IShowFormat>,
			default: { noChecked: '${total}', hasChecked: '${checked}/${total}' },
		},
		widths: {
			type: Array, //as PropType<ObjectConstructor>,
			default: () => [300, 300],
		},
		titles: {
			type: Array, //as PropType<ObjectConstructor>,
			default: () => [' ', ' '],
		},
		buttonTexts: {
			type: Array, //as PropType<ObjectConstructor>,
			default: () => ['', ''],
		},
		emptyTexts: {
			type: Array, //as PropType<ObjectConstructor>,
			default: () => ['No Data', 'No Data'],
		},
		saveSvc: {
			type: Function as PropType<saveAssociatedSvc>,
			required: true,
		},
		getDataSvc: {
			type: Function as PropType<getAssociatedSvc>,
			required: true,
		},
		createOption: {
			type: Function as PropType<createOption>,
			default: (item: ITransferItem) => {
				return {
					key: item.id,
					label: `${item.name}`,
					disabled: false,
					data: item,
				};
			},
		},
	},
	emits: {
		//@ts-ignore
		saved: (data: any) => true,
	},
	directives: {
		// 局部注册 v-loading 指令
		['loading']: vLoading,
	},
	setup(prop, ctx) {
		const diaglogForm = ref<DialogFormInstance>();
		const DATA = reactive<
			FormData<number, IAssociation> & {
				rawSelectValue: Array<number | string>;
				loading: boolean;
				data: Option[];
			}
		>({
			operation: FormOperation.add,
			id: 0,
			loading: false,
			data: [],
			rawSelectValue: [],
			fromData: {
				add: [],
				remove: [],
			},
		});
		//临时存储
		const selectValue = ref<Array<number | string>>([]);
		//@ts-ignore
		const key = Symbol('formData') as InjectionKey<FormItem>; //'formData'
		provide('formData', DATA.fromData);
		const saveBefore = () => {
			const ids = selectValue.value;
			DATA.fromData.add = minus(ids, DATA.rawSelectValue);
			DATA.fromData.remove = minus(DATA.rawSelectValue, ids);
		};
		const save = () => {
			//提交数据
			let promist: Promise<IResponse>;
			promist = prop.saveSvc(DATA.id, DATA.fromData);
			showLoading();
			promist
				.then((res) => {
					diaglogForm.value?.closeDialog();
					ElMessage.success(res.message || `操作成功`);
					ctx.emit('saved', res);
				})
				.finally(() => {
					closeLoading();
				});
		};

		onMounted(async () => {});
		onBeforeUnmount(() => {});

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
		const rules: FormRules = {
			selectData: [
				{ required: true, validator: onValidator, trigger: ['blur', 'change'] },
			],
		};
		const onChange = (keyList: Array<Object>) => {
			//console.log(keyList)
			saveBefore();
		};

		const slots = useSlots();
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
					<style>{` 
						#transferContent .el-transfer .el-transfer-panel:first-child { width:${prop.widths[0]}px;}
						#transferContent  .el-transfer .el-transfer-panel:last-child { width: ${prop.widths[1]}px; }
						#transferContent  .el-form-item__content { margin-left:0px !important;}
					`}</style>
					<ElFormItem prop="selectData" id="transferContent">
						<ElTransfer
							vLoading={DATA.loading}
							v-model={selectValue.value}
							data={DATA.data}
							format={prop.format}
							filterable={prop.filterable}
							titles={prop.titles}
							onChange={onChange}
							buttonTexts={prop.buttonTexts}>
							{{
								default: (data: { option: Option }) => (
									<ElTooltip
										showAfter={prop.afterShowTip}
										content={`${data.option.label}`}>
										<span>{`${data.option.label}`}</span>
									</ElTooltip>
								),
								'left-footer': () => slots.leftFooter && slots.leftFooter(),
								'right-footer': () => slots.rightFooter && slots.rightFooter(),
								'left-empty': () =>
									(slots.leftEmpty && slots.leftEmpty()) || (
										<ElEmpty imageSize={60} description={prop.emptyTexts[0]} />
									),
								'right-empty': () =>
									(slots.rightEmpty && slots.rightEmpty()) || (
										<ElEmpty imageSize={60} description={prop.emptyTexts[1]} />
									),
							}}
						</ElTransfer>
					</ElFormItem>
				</>
			),
		};

		const rander = (): VNode => {
			return (
				<DialogForm
					title={prop.title}
					style={ctx.attrs}
					rules={rules}
					closeOnClickModal={false}
					draggable
					ref={diaglogForm}
					v-slots={fromSlots}
				/>
			);
		};
		const openDialog = (id: number) => {
			DATA.loading = true;
			DATA.id = id;
			prop
				.getDataSvc(id, {})
				.then((res) => {
					const data: Option[] = [];
					selectValue.value = [];
					DATA.rawSelectValue = [];
					res.data.forEach((item: ITransferItem) => {
						if (item.associatedValue)
							selectValue.value.push(item.id),
								DATA.rawSelectValue.push(item.id);
						data.push(prop.createOption(item));
					});
					DATA.data = data;
				})
				.finally(() => {
					DATA.loading = false;
				});
			diaglogForm.value?.openDialog();
		};
		ctx.expose({
			openDialog,
		});
		rander.openDialog = openDialog;
		return rander;
	}, //end setup
});
