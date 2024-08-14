import { RawAxiosRequestHeaders } from 'axios'; //AxiosRequestHeaders,
export const INSTALLED_KEY = Symbol('co6co-right');

export interface IDownloadConfig {
	method?: string;
	headers?: RawAxiosRequestHeaders & {
		Range: string /* `bytes=${start}-${end}`*/;
	};
}

export enum Flag {
	Y = 'Y',
	N = 'N',
}

export type validatorBack = (error?: string | Error | undefined) => void;
