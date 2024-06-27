export interface IResponse<T = any> {
	code: number;
	message: string;
	data: T extends any ? T : T & any;
}
export interface IPageResponse<T = any> extends IResponse<T> {
	total: number;
}
export interface IPageParam<T = any> {
	pageIndex: number;
	pageSize: number;
	orderBy?: string;
	order?: 'desc' | 'asc'; // string //[desc|asc]
	data?: T extends any ? T : T & any;
}
export interface Table_Module_Base {
	pageTotal: number;
	diaglogTitle?: string;
}

/**
 * 增|编 表单所有模块
 */
export interface FormData<TKey, T> {
	operation: FormOperation;
	id: TKey;
	fromData: T;
}
export interface FormItemBase {
	createTime: string;
	updateTime: string;
	createUser: number;
	updateUser: number;
}

/**
 * @deprecated 将于下个版本 0.0.3 被弃用
 * //下拉框
 *  功能与 ISelect 相似
 */
export interface SelectItem {
	id: number;
	name: string;
}

//树形选择
export interface ITreeSelect {
	id: number;
	name: string;
	parentId: number;
	children?: ITreeSelect[];
}

export interface IAssociation {
	add: Array<number | string>;
	remove: Array<number | string>;
}

export interface ISelect {
	id: number | bigint;
	name: string;
}
export interface IEnumSelect {
	uid: number | bigint; //  Enum key
	key: string;
	label: string;
	value: number | string;
}

export enum FormOperation {
	add,
	edit,
}
export enum Operation {
	Add,
	Edit,
	Del,
}
export type ObjectType = { [key: string]: any };
export type Direction = 'vertical' | 'horizontal';
export type ElTagType = 'success' | 'info' | 'warning' | 'danger' | '';

export const getEleTagTypeByBoolean = (v: number | boolean) => {
	let isSuccess = true;
	if (typeof v == 'number') isSuccess = Boolean(v);
	else isSuccess = v;
	if (isSuccess) return 'success';
	return 'danger';
};

export const tree_props = {
	value: 'id',
	label: 'name',
	children: 'children',
};
