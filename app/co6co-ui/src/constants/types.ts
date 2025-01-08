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
export type FormItemBase2 = Pick<FormItemBase, 'createTime' | 'updateTime'>;

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
export interface ITree {
	children?: ITree[];
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

export interface IAuthonInfo {
	token: string;
	expireSeconds: number;
	sessionId: string;
	refreshToken: {
		token: string;
		expireSeconds: number;
	};
}

export type Point = {
	lng: number; // 经度
	lat: number; // 纬度
};

export enum HttpContentType {
	json = 'application/json;charset=utf-8',
	multipart = 'multipart/form-data',
	form = 'x-www-form-urlencoded;charset=UTF-8',
	text = 'text/plain',
	html = 'text/html',
	image = 'image/jpeg', //image/jpeg, image/png, image/gif
	xml = 'application/xml',
	stream = 'application/octet-stream',
	/**
	 *  video/mp4：MP4 格式的视频文件。
		video/quicktime：QuickTime 格式的视频文件（.mov）。
		video/mpeg：MPEG 视频文件。
		video/x-msvideo 或 video/avi：AVI 视频文件。
		video/x-flv：FLV (Flash Video) 文件。
		video/webm：WebM 视频文件，一种开放的多媒体容器格式。
		video/3gpp 或 video/3gp：用于移动设备的 3GPP 格式视频文件。
		video/x-matroska 或 video/x-mkv：MKV (Matroska) 视频文件。
		application/vnd.ms-asf 或 video/x-ms-wmv：WMV (Windows Media Video) 文件
	 */
	video = 'video/mp4',
}
