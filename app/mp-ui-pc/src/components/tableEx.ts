import {
	ref,
	watch,
	reactive,
	nextTick,
	PropType,
	onMounted,
	onBeforeUnmount,
	computed,
} from 'vue';

export  interface PagedOption {
	pageIndex: number;
	pageTotal: number;
	pageSizes: Array<number>;
	pageSize: number;
	layout: String; //total, sizes,  prev, pager, next, jumper ->
}
export const pagedOption = reactive<PagedOption>({
	pageIndex: 1,
	pageTotal: 0,
	pageSizes: [10, 15, 20, 30, 50, 100, 200, 300, 500, 1000],
	pageSize: 15,
	layout: 'prev, pager, next,total,jumper',
});
/*
const keyDown = (e: KeyboardEvent) => {
	if (e.ctrlKey) {
		if (['ArrowLeft', 'ArrowRight'].indexOf(e.key) > -1) {
			let current = pagedOption.pageIndex.valueOf();
			let v = e.key == 'ArrowRight' || e.key == 'd' ? current + 1 : current - 1;
			onPageChange(v);
		}
		if (['ArrowUp', 'ArrowDown'].indexOf(e.key) > -1) {
			let current = currentTableItemIndex.value;
			if (!current) current = 0;
			let v = e.key == 'ArrowDown' || e.key == 's' ? current + 1 : current - 1;
			if (0 <= v && v < tableInstance._value.data.length) {
				setTableSelectItem(v);
			} else {
				if (v < 0) ElMessage.error('已经是第一条了');
				else if (v >= tableInstance._value.data.length)
					ElMessage.error('已经是最后一条了');
			}
		}
	}
	//process_view.value.keyDown(e)
	e.stopPropagation();
};
*/
