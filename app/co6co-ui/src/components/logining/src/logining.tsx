import {
	// createElementBlock,
	//createElementVNode,
	defineComponent,
	//openBlock,
	watchEffect,
} from 'vue';
import { ElLoading } from 'element-plus';
/* 全局请求 loading */
let loadingInstance: ReturnType<typeof ElLoading.service>;
const start = () => {
	loadingInstance = ElLoading.service({
		fullscreen: true,
		lock: true,
		text: 'Loading',
		background: 'rgba(0, 0, 0, 0.2)',
	});
};
const close = () => {
	loadingInstance.close();
};

let needLoadingRequestCount = 0;
export const showLoading = () => {
	if (needLoadingRequestCount === 0) {
		start();
	}
	needLoadingRequestCount++;
};
export const closeLoading = () => {
	if (needLoadingRequestCount <= 0) return;
	needLoadingRequestCount--;
	if (needLoadingRequestCount === 0) {
		close();
	}
};
//全屏loging
export default defineComponent({
	name: 'Logining',
	props: {
		logining: {
			type: Boolean,
			default: true,
		},
	},
	setup(prop) {
		const _loading = ElLoading.service({ fullscreen: true });
		watchEffect(() => {
			if (!prop.logining) _loading.close();
			_loading.visible.value = true;
		});
		return () => <></>;
	},
});
