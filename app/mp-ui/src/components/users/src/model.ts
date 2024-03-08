interface userState {
	list: Array<{ key: String; label: String; value: number }>;
	getStateName: (state: number) => String;
}

const state = {
	list: [
		{ key: '启用', label: '启用', value: 0 },
		{ key: '锁定', label: '锁定', value: 1 },
		{ key: '禁用', label: '禁用', value: 2 },
	],
	getStateName: (v: number) => {  
		return state.list.find((m) => m.value == v);
	},
};

export { state };
