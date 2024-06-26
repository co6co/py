import {
	computed,
	defineComponent,
	onMounted,
	onUnmounted,
	reactive,
} from 'vue';
import { ElMenu, ElMenuItem } from 'element-plus';

import { type ObjectType } from '../../../constants';
import type { CSSProperties } from 'vue';

export interface ContextMenuItem {
	id: number | string;
	name: number | string;
}
export interface IContextMenu {
	visible: boolean;
	left: number;
	top: number;
	data: ContextMenuItem[];
	context?: any;
}
export default defineComponent({
	name: 'EcContextMenu',
	emits: {
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		// @ts-ignore
		checked: (index: number, item: ContextMenuItem, context?: any) => true,
	},
	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	// @ts-ignore
	setup(prop, { attrs, slots, emit, expose }) {
		const menuData = reactive<IContextMenu>({
			visible: false,
			left: 0,
			top: 0,
			data: [],
		});
		const open = (
			data: ContextMenuItem[],
			event: MouseEvent,
			context?: any
		) => {
			event.preventDefault(); //阻止鼠标右键默认行为
			menuData.data = data;
			menuData.left = event.clientX;
			menuData.top = event.pageY;
			menuData.visible = true;
			menuData.context = context;
		};

		const onSelectMenu = (index: number, item: ContextMenuItem) => {
			menuData.visible = false;
			emit('checked', index, item, menuData.context);
		};
		const style = computed(() => {
			return {
				left: `${menuData.left}px`,
				top: `${menuData.top}px`,
				position: 'fixed',
				'z-index': 9,
			};
		});
		const close = () => {
			menuData.visible = false;
		};
		onMounted(() => {
			document.addEventListener('click', close);
		});
		onUnmounted(() => {
			document.removeEventListener('click', close);
		});
		const render = (): ObjectType => {
			//可以写某些代码
			return (
				<>
					{menuData.visible ? (
						<div style={style.value as CSSProperties}>
							<ElMenu mode="vertical">
								{menuData.data.map((item, index) => {
									return (
										<ElMenuItem
											style="height:32px;line-height:32px;border-bottom: 1px solid var(--el-border-color);"
											index={index.toString()}
											onClick={() => onSelectMenu(index, item)}>
											{item.name}
										</ElMenuItem>
									);
								})}
							</ElMenu>
						</div>
					) : (
						<></>
					)}
				</>
			);
		};

		expose({
			open,
		});
		render.open = open;

		return render;
	},
});
