import { type SlotsType, defineComponent, reactive, VNode } from 'vue';
import { ElButton, ElDialog, ElScrollbar } from 'element-plus';

import style from '@/assets/css/dialog.module.less';
export interface IDialogDataType {
	visible: boolean;
	title?: string;
}
/**
 * provide 主组件使用
 * inject  子组件使用
 */

export default defineComponent({
	name: 'EcDialog',
	props: {
		title: {
			type: String,
			default: '弹出框',
		},
		closeTxt: {
			type: String,
			default: '关闭',
		},
	},

	//定义事件类型
	emits: {
		//事件名可以使用字符
		close: () => true,
	},
	slots: Object as SlotsType<{
		default: () => any;
		buttons: () => any;
	}>,

	setup(prop, ctx) {
		/*
    '''
    // props.data
    // ctx.attrs    ctx.slots    ctx.emit 
    '''
    */
		const diaglogData = reactive<IDialogDataType>({
			visible: false,
		});
		//其他api 操作
		//end

		const onOpenDialog = () => {
			diaglogData.visible = true;
		};

		const dialogSlots = {
			footer: () => {
				return (
					<span class="dialog-footer">
						<ElButton
							onClick={() => {
								diaglogData.visible = false;
								ctx.emit('close');
							}}
							v-slots={{ default: () => prop.closeTxt }}
						/>
						{ctx.slots.buttons ? ctx.slots.buttons() : null}
					</span>
				);
			},
		};
		const rander = (): VNode => {
			/**
			 * 还能在该位置写一些计算每次状态改变都能被渲染
			 */
			return (
				<ElDialog
					class={style['diaglog-box']}
					title={prop.title}
					v-model={diaglogData.visible}
					v-slots={dialogSlots}>
					<ElScrollbar>
						<>{ctx.slots.default ? ctx.slots.default() : null}</>
					</ElScrollbar>
				</ElDialog>
			);
		};
		//必须导出不然运行时就 没有了
		const expose = {
			openDialog: onOpenDialog,
			data: diaglogData,
		};
		ctx.expose(expose);
		//defineExpose( {openDialog: onOpenDialog})  //在 tsx  不起作用

		/*
		Object.keys(expose).forEach((key, index, keys) => {
		console.info("Keys",key,index)
		let o:ObjectType=expose  
		rander.openDialog=o[key]
		})
		*/

		//为了让ts能检测到
		rander.openDialog = expose.openDialog;
		rander.data = diaglogData;
		//有模板的不能返回对象只能是 Function
		return rander;
	}, //end setup
});
