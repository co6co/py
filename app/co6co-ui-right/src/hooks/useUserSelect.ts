import { get_state_svc, get_category_svc } from '@/api/sys/user';
import { useEnum } from './base';
import { useStore } from 'co6co';
import { get_user_avatar } from '@/api/sys/user';
import { create_URL_resource } from '@/api/download';
import { ref } from 'vue';
export const useState = () => {
	const { loadData, selectData, refresh, getName } = useEnum(get_state_svc);
	const getTagType = (value?: number) => {
		if (value != undefined) {
			switch (value) {
				case 0:
					return 'success';
				case 1:
					return 'danger';
				case 2:
					return 'warning';
			}
		}
		return 'info';
	};

	return { loadData, selectData, refresh, getName, getTagType };
};

export const useCategory = () => {
	const { loadData, selectData, refresh, getName } = useEnum(get_category_svc);
	return { loadData, selectData, refresh, getName };
};

/**
 * 需要分析下为什么useStore 不能用
 * @returns
 */
export const getUserInfo = () => {
	const store = useStore(); //todo 不能用store
	const avatar = ref();
	const updateAvatar = async () => {
		const res = await get_user_avatar();
		if (res) {
			store.setConfig('avatar', create_URL_resource({ data: res.data }));
			avatar.value = create_URL_resource({ data: res.data });
		}
	};

	const getAvatar = (): string => {
		return avatar.value;
		//return store.getConfig('avatar');
	};

	updateAvatar();
	return { avatar, getAvatar, updateAvatar };
};
