import { ref, watch } from 'vue';

/**
 * 用于处理 modelValue 的封装
 *  <input v-model="localValue" @input="onChange">
 * const { localValue, onChange } = useModelWrapper(props, ctx);
 *
 * @param props 组件的 props
 * @param ctx 组件的上下文对象，包含 emit 方法
 *
 **/
export function useModelWrapper(props, ctx, watchBck?: (newValue) => void) {
	const localValue = ref(props.modelValue);
	watch(
		() => props.modelValue,
		(newValue) => {
			localValue.value = newValue;
			// 这里的 ... 部分你可以根据实际需求补充具体逻辑
			if (watchBck) {
				watchBck(newValue);
			}
		}
	);
	const onChange = (newValue) => {
		localValue.value = newValue;
		ctx.emit('update:modelValue', newValue);
	};
	return {
		localValue,
		onChange,
	};
}
