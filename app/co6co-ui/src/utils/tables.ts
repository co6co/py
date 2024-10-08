import { type IPageParam } from '../constants';

/**
 * 获取table 行索引号
 *
 * @param param 查询参数
 * @param scopeIndex
 * @returns
 */
export const getTableIndex = (param: IPageParam, scopeIndex: number) =>
	(param.pageIndex - 1) * param.pageSize + scopeIndex + 1;

/**
 * 排序
 *
 * @param column 选中的列
 * @param param 查询参数
 * @param bck 回调函数 通常式调用API 的方法
 */
export const onColChange = (column: any, param: IPageParam, bck: any) => {
	param.order = column.order === 'descending' ? 'desc' : 'asc';
	param.orderBy = column.prop;
	if (column && bck) bck(); // 获取数据的方法
};
