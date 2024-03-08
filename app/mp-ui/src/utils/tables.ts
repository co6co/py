//获取所有号
export const getTableIndex = (param: IpageParam, scopeIndex: number) =>
	(param.pageIndex - 1) * param.pageSize + scopeIndex + 1;
