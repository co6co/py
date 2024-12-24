import { ref } from 'vue';

import { get_config_svc, add_svc, type IConfig } from '@/api/config/svcs';
export const useConfig = (code: string) => {
	const config = ref<IConfig>();
	const refresh = async () => {
		//console.info(`获取配置${code}`);
		const res = await get_config_svc(code);
		config.value = res.data;
	};
	/**
	 *
	 * @param noconfigAppend 配置中没有自动增加配置
	 * @returns
	 */
	const getValue = (noconfigAppend?: boolean) => {
		let result = config.value?.value;
		if (!result) {
			result = '/';
			if (noconfigAppend) {
				const param = {
					code: code,
					dictFlag: 'N',
					name: code + '_config',
					sysFlag: 'N',
					value: result,
				};
				add_svc(param).then((res) => {});
			}
		}
		return result;
	};
	const loadData = refresh;
	return { loadData, config, refresh, getValue };
};
