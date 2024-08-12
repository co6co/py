import { createServiceInstance } from 'co6co';
import * as api_type from 'co6co';

export const create_svc = (baseUrl: string) => {
	const get_select_svc = (
		data?: any
	): Promise<api_type.IResponse<api_type.ISelect[]>> => {
		return createServiceInstance().get(`${baseUrl}`, data);
	};
	const get_table_svc = (data: any): Promise<api_type.IPageResponse> => {
		return createServiceInstance().post(`${baseUrl}`, data);
	};
	/*
	const exist_svc = (code: string): Promise<api_type.IResponse> => {
		return createServiceInstance().post(`${baseUrl}/exist/${code}`);
	};
	*/
	const exist_svc = (
		code: string,
		id: number = 0
	): Promise<api_type.IResponse<boolean>> => {
		return createServiceInstance().get(`${baseUrl}/exist/${code}/${id}`);
	};
	const add_svc = (data: any): Promise<api_type.IResponse> => {
		return createServiceInstance().put(`${baseUrl}`, data);
	};
	const edit_svc = (id: number, data: any): Promise<api_type.IResponse> => {
		return createServiceInstance().put(`${baseUrl}/${id}`, data);
	};
	const del_svc = (
		id: number,
		reason?: string
	): Promise<api_type.IResponse> => {
		let url = `${baseUrl}/${id}`;
		if (reason) url = `${baseUrl}/${id}?reason=${reason}`;
		return createServiceInstance().delete(url);
	};
	return {
		get_select_svc,
		get_table_svc,
		exist_svc,
		add_svc,
		edit_svc,
		del_svc,
	};
};
export const create_tree_svc = (baseUrl: string) => {
	const get_select_tree_svc = (
		key?: number | string
	): Promise<api_type.IPageResponse> => {
		if (key != undefined)
			return createServiceInstance().get(`${baseUrl}/tree/${key}`);
		else return createServiceInstance().get(`${baseUrl}/tree`);
	};
	const get_tree_table_svc = (data: any): Promise<api_type.IPageResponse> => {
		return createServiceInstance().post(`${baseUrl}/tree`, data);
	};
	const service = create_svc(baseUrl);
	return { get_select_tree_svc, get_tree_table_svc, ...service };
};

export const create_association_svc = (baseUrl: string) => {
	const get_association_svc = (
		associationValue: number | string,
		data: any
	): Promise<api_type.IResponse> => {
		return createServiceInstance().post(
			`${baseUrl}/association/${associationValue}`,
			data
		);
	};
	const save_association_svc = (
		associationValue: number | string,
		data: api_type.IAssociation
	): Promise<api_type.IResponse> => {
		return createServiceInstance().put(
			`${baseUrl}/association/${associationValue}`,
			data
		);
	};
	return { get_association_svc, save_association_svc };
};
