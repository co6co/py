import { defineComponent, PropType, VNodeChild, SlotsType } from 'vue';
import { ref, reactive, onMounted } from 'vue';
import {
	ElTable,
	ElContainer,
	ElMain,
	ElHeader,
	ElScrollbar,
	ElFooter,
} from 'element-plus';
import {
	showLoading,
	closeLoading,
	type IPageParam,
	onColChange,
	type Table_Module_Base,
	Pagination,
	IPageResponse,
	getTableIndex,
} from 'co6co';
type DataApi = (query: any) => Promise<IPageResponse>;
type filter = (data: any) => Array<Object>;
type beforeApi = (queryData: any) => any;
type afterApi = () => void;
export default defineComponent({
	name: 'tableView',
	inheritAttrs: false,
	props: {
		dataApi: {
			type: Function as PropType<DataApi>,
			required: true,
		},
		beforeApi: {
			type: Function as PropType<beforeApi>,
			required: false,
		},
		afterApi: {
			type: Function as PropType<afterApi>,
			required: false,
		},
		query: {
			type: Object,
			default: {},
		},
		resultFilter: {
			type: Function as PropType<filter>,
			default: undefined,
		},
		showPaged: {
			type: Boolean,
			default: true,
		},
		autoLoadData: {
			type: Boolean,
			default: true,
		},
	},
	slots: Object as SlotsType<{
		header: () => any;
		default: () => any;
		footer: () => any;
	}>,
	setup(prop, ctx) {
		//:define
		interface IQueryItem extends IPageParam {
			appid?: string;
			title?: string;
		}
		interface Table_Module extends Table_Module_Base {
			query: IQueryItem;
			data: any[];
			currentItem?: any;
		}
		//:use
		//const { getPermissKey } = routeHook.usePermission()
		//const dictHook = useDictHook.useDictSelect()
		//const store = get_store()
		//end use
		//:page
		const tableRef = ref<InstanceType<typeof ElTable>>();
		const DATA = reactive<Table_Module>({
			query: {
				pageIndex: 1,
				pageSize: 10,
				order: 'asc',
				orderBy: '',
			},
			data: [],
			pageTotal: -1,
			diaglogTitle: '',
		});
		// 获取表格数据
		const queryData = () => {
			showLoading();
			let data = prop.showPaged
				? { ...DATA.query, ...prop.query }
				: { ...prop.query };
			if (prop.beforeApi) data = prop.beforeApi(data);
			prop
				.dataApi(data)
				.then((res) => {
					if (prop.resultFilter) DATA.data = prop.resultFilter(res.data);
					else DATA.data = res.data;
					DATA.pageTotal = res.total || -1;
				})
				.finally(() => {
					closeLoading();
					if (prop.afterApi) prop.afterApi();
				});
		};
		const refesh = () => {
			queryData();
		};
		const search = () => {
			DATA.query.pageIndex = 1;
			queryData();
		};
		const onColChange2 = (column: any) => {
			onColChange(column, DATA.query, queryData);
		};
		const onRowChanged = (row) => {
			DATA.currentItem = row;
		};
		//end page
		onMounted(async () => {
			if (prop.autoLoadData) queryData();
		});
		const rowIndex = ($index: number) => {
			return getTableIndex(DATA.query, $index);
		};
		//:page reader
		const rander = (): VNodeChild => {
			return (
				<div class="container-layout c-container">
					<ElContainer>
						<ElHeader>{ctx.slots.header?.()}</ElHeader>
						<ElMain>
							<ElScrollbar>
								<ElTable
									{...ctx.attrs}
									data={DATA.data}
									border={true}
									class="table"
									ref={tableRef}
									headerCellClassName="table-header"
									onRow-click={onRowChanged}
									onSort-change={onColChange2}>
									{ctx.slots.default?.()}
								</ElTable>
							</ElScrollbar>
						</ElMain>
						{prop.showPaged ? (
							<ElFooter>
								{/*增加Pagination的相关参数*/}
								<Pagination
									option={DATA.query}
									total={DATA.pageTotal}
									onCurrentPageChange={queryData}
									onSizeChage={queryData}
								/>
							</ElFooter>
						) : (
							<></>
						)}
					</ElContainer>
					{ctx.slots.footer?.()}
				</div>
			);
		};

		ctx.expose({
			refesh,
			search,
			rowIndex,
			tableRef,
		});
		rander.refesh = refesh;
		rander.search = search;
		rander.rowIndex = rowIndex;
		rander.tableRef = tableRef;

		return rander;
	}, //end setup
});
