export * from './route';
import { nextTick } from 'vue';
import { ElScrollbar, ElTable } from 'element-plus';
/**
 * 滚动Table到指定行
 * @param rowIndex  table 行
 * @param table
 * @param scrollby
 */
export const scrollTableToRow = (
	rowIndex: number,
	table: InstanceType<typeof ElTable>,
	scrollby: InstanceType<typeof ElScrollbar>
) => {
	nextTick(() => {
		if (table) {
			// 获取目标行的 DOM 元素
			//tableRef?.bodyWrapper.querySelector
			const targetRow = table.$el.querySelector(
				`tbody tr:nth-child(${rowIndex + 1})`
			);
			if (targetRow) {
				// 计算目标行相对于 el-scrollbar.wrap 的位置
				const targetOffsetTop = targetRow.offsetTop;
				// 设置滚动条的位置
				scrollby.setScrollTop(targetOffsetTop);
			}
		}
	});
};
