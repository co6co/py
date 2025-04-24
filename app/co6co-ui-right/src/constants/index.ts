import { RawAxiosRequestHeaders } from 'axios'; //AxiosRequestHeaders,
export const INSTALLED_KEY = Symbol('co6co-right');

export interface IDownloadConfig {
	method?: string;
	timeout?: number;
	headers?: RawAxiosRequestHeaders & {
		Range: string /* `bytes=${start}-${end}`*/;
	};
}

export enum Flag {
	Y = 'Y',
	N = 'N',
}

export type validatorBack = (error?: string | Error | undefined) => void;

/**
object：
	当你希望确保一个值是一个对象（非原始类型 number、string、boolean、symbol、null 和 undefined）时。
	当你不想允许 null 或 undefined 时。
object 更严格，只表示非原始类型对象，不包括 null 和 undefined。
Object 更宽泛，可以表示任何非原始类型，包括 null 和 undefined。

number 是一个原始值，直接表示数值。
Number 是一个对象，是对 number 原始值的包装。可以调用一些方法，如 toFixed()、toExponential() 等。
 */
/**
 * ElTable Row 对象
 */
export interface tableScope<T = object> {
	$index: number;
	cellIndex: number;
	expanded: boolean;
	row: T;
}

/**
 * 客户端选择文件
 * 使用
 */
export interface IFileOption {
	file: File;
	percentage?: number;
	subPath?: String;
	finished?: boolean;
}
export enum DictShowCategory {
	NameValueFlag = 0, //默认
	NameValue = 1,
	NameFlag = 2,
	All = 999,
}

export * from './api';
