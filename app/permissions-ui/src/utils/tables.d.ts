/**
 * 获取table 行索引号
 *
 * @param param 查询参数
 * @param scopeIndex
 * @returns
 */
export declare const getTableIndex: (param: IPageParam, scopeIndex: number) => number;
/**
 * 排序
 *
 * @param column 选中的列
 * @param param 查询参数
 * @param bck 回调函数 通常式调用API 的方法
 */
export declare const onColChange: (column: any, param: IPageParam, bck: any) => void;
/**
 * 分页
 *
 * @param val 当前页
 * @param param 查询参数
 * @param bck 回调
 */
export declare const onPageChange: (val: number, param: IPageParam, bck: any) => void;
