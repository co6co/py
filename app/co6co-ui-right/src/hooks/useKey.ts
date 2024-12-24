import { onMounted, onUnmounted } from 'vue';
export const useKeyUp = (bck: (event: KeyboardEvent) => void) => {
	// 定义处理 Esc 键按下的方法
	const handleEsc = (event: KeyboardEvent) => {
		bck(event);
	};

	// 在组件挂载时添加全局键盘事件监听器
	onMounted(() => {
		window.addEventListener('keyup', handleEsc);
	});

	// 在组件卸载时移除监听器，防止内存泄漏
	onUnmounted(() => {
		window.removeEventListener('keyup', handleEsc);
	});
};
