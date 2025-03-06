import { createServiceInstance, type IResponse } from 'co6co';
const base_URL = '/api/verify';

export interface IDragVerifyData {
	start: number;
	data: Array<{ t: number; x: number; y: number }>;
	end: number;
}
export const dragVerify_Svc = (
	data: IDragVerifyData
): Promise<IResponse<string>> => {
	return createServiceInstance(3000, false).post(`${base_URL}`, data);
};
