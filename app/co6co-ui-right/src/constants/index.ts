import { AxiosRequestHeaders } from 'axios';
export const INSTALLED_KEY = Symbol('co6co-right');

export interface IDownloadConfig {
	method?: string;
	headers?: AxiosRequestHeaders & {
		Range: string /* `bytes=${start}-${end}`*/;
	};
}

export enum Flag {
	Y = 'Y',
	N = 'N',
}
