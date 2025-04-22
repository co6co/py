const PageDefaultLayouts = ['prev', 'pager', 'next', 'total', 'sizes'] as const;
const PageLayouts = ['prev', 'pager', 'next', 'total', 'sizes'] as const;
const PageAllLayouts = [
	'prev',
	'pager',
	'next',
	'jumper',
	'total',
	'sizes',
] as const;
export { PageDefaultLayouts, PageLayouts, PageAllLayouts };
export interface IPageParam<T = any> {
	pageIndex: number;
	pageSize: number;
	orderBy?: string;
	order?: 'desc' | 'asc'; // string //[desc|asc]
	data?: T extends any ? T : T & any;
}
