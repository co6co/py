import { defineComponent, VNodeChild, PropType, computed, reactive } from 'vue';
import { ElPagination } from 'element-plus';
import { type IPageParam } from '@/constants';

//使用 `as const` 来确保 `allowedColors` 是一个不可变的数组
const allowedLayouts = [
	'prev',
	'pager',
	'next',
	'jumper',
	'total',
	'sizes',
] as const;
type AllowedLayouts = (typeof allowedLayouts)[number];

export const PaginationProps = {
	background: {
		type: Boolean,
		default: true,
	},
	layouts: {
		type: Object as PropType<Array<AllowedLayouts>>, // 使用类型断言指定类型
		default: ['prev', 'pager', 'next', 'total'],
	},
	pageSizes: {
		type: Object as PropType<Array<number>>, // 使用类型断言指定类型
		default: [10, 20, 30, 40, 50, 100],
	},
} as const;

//定义属性
const props = {
	...PaginationProps,
	/**
	 *属性串进去
	 */
	option: {
		type: Object as PropType<IPageParam>,
		required: true,
	},
	total: {
		type: Number,
		required: true,
	},
} as const;

//定义事件
//没什么用 方法编译出来的方法太泛
/*
interface Emits {
	(e: 'sizeChage', value: number): void;
	(e: 'currentPageChange', value: number): void;
}*/

export default defineComponent({
	props: props, //里面的值不能直接在本模块中修改
	//emits: ['sizeChage', 'currentPageChange'] as const, // 这里的 as const 是为了生成正确的类型定义
	emits: {
		sizeChage: (value: number) => true,
		currentPageChange: (value: number) => true,
	},
	setup(prop, { emit }) {
		//:define

		//:use
		//end use

		//:page
		const layout = computed(() => {
			return prop.layouts.join();
		});
		//end page
		const DATA = reactive({
			query: prop.option,
			// total: prop.total
		});
		const onSizeChage = (val: number) => {
			DATA.query.pageSize = val;
			emit('sizeChage', val);
		};
		const onCurrentChange = (val: number) => {
			DATA.query.pageIndex = val;
			emit('currentPageChange', val);
		};
		// 当属性为基础对象时
		/*
    watch(
      () => prop.total,
      (n) => {
        DATA.total = prop.total
      }
    )*/

		//:page reader
		const rander = (): VNodeChild => {
			return (
				<>
					<ElPagination
						background={prop.background}
						pageSize={DATA.query.pageSize}
						pageSizes={prop.pageSizes}
						total={prop.total}
						currentPage={DATA.query.pageIndex}
						onSizeChange={onSizeChage}
						onCurrentChange={onCurrentChange}
						layout={layout.value}
					/>
				</>
			);
		};
		return rander;
	}, //end setup
});
