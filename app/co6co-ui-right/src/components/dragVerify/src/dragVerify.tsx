// DragVerify.tsx
import { ElButton } from 'element-plus';
import { defineComponent, ref, reactive, onMounted, onUnmounted } from 'vue';
import style from '@/assets/css/dragVerify.module.less';
import { IDragVerifyData, dragVerify_Svc } from '@/api/verify';
import { ArrowRight } from '@element-plus/icons-vue';

import { useModelWrapper, isMobileBrowser } from 'co6co';
const DragVerify = defineComponent({
	props: {
		width: {
			type: Number,
			default: 300,
		},
		height: {
			type: Number,
			default: 50,
		},
		successText: {
			type: String,
			default: undefined,
		},
		text: {
			type: String,
			default: '拖动滑块完成验证',
		},
		onVerifySuccess: {
			type: Function,
			required: true,
		},
		modelValue: {
			type: String,
		},
	},
	emits: {
		'update:modelValue': (data: String) => true,
	},
	setup(props, ctx) {
		const sliderRef = ref<InstanceType<typeof ElButton> | null>(null);
		const BgRef = ref<HTMLDivElement | null>(null);
		const bgWidth = ref(0);
		const isDragging = ref(false);
		const offsetX = ref(0);
		const currentX = ref(0);
		const verifySuccess = ref(false);
		const message = ref('');

		const DATA = reactive<IDragVerifyData>({
			start: 0,
			data: [],
			end: 0,
		});
		//存储本地值
		//const localValue = ref(props.modelValue);
		// 监听 modelValue 的变化 更新本地值
		//watch(
		//	() => props.modelValue,
		//	(newValue) => {
		//		localValue.value = newValue;
		//		//需要重新验证时
		//		if (!newValue) {
		//			currentX.value = 0;
		//			verifySuccess.value = false;
		//		}
		//	}
		//);

		//const onChange = (newValue: string) => {
		//	localValue.value = newValue;
		//	ctx.emit('update:modelValue', newValue);
		//};
		const { onChange } = useModelWrapper(props, ctx, (newValue) => {
			if (!newValue) {
				currentX.value = 0;
				verifySuccess.value = false;
			}
		});
		const isMobile = ref(false);
		const handleMouseDown = (e: MouseEvent) => {
			onstart(e.clientX);
		};
		const onstart = (clientX) => {
			if (verifySuccess.value) return;
			isDragging.value = true;
			offsetX.value = clientX;
			DATA.start = new Date().getTime();
		};
		const onMove = (e: { clientX; clientY }, onStop: () => void) => {
			const diffX = e.clientX - offsetX.value;
			DATA.data.push({
				t: new Date().getTime(),
				x: e.clientX,
				y: e.clientY,
			});
			currentX.value = Math.min(
				Math.max(0, diffX),
				props.width - (sliderRef.value?.$el.offsetWidth || 0)
			);
			if (
				currentX.value >=
				bgWidth.value - (sliderRef.value?.$el.offsetWidth || 0)
			) {
				DATA.end = new Date().getTime();
				verifySuccess.value = true;
				//停止
				if (onStop) onStop();
				dragVerify_Svc(DATA)
					.then((res) => {
						onChange(res.data);
						message.value = res.message;
						props.onVerifySuccess(res.data, DATA.start, DATA.end);
					})
					.catch((err) => {
						verifySuccess.value = false;
						currentX.value = 0;
						//不是预期的错误打印日志
						if (!err.code) console.log(err);
					})
					.finally(() => {
						DATA.data = [];
					});
			}
		};

		const handleMouseMove = (e: MouseEvent) => {
			//console.log('handleMouseMove', e);
			// 取消上一次的requestAnimationFrame请求
			if (!isDragging.value || verifySuccess.value) return;
			onMove({ clientX: e.clientX, clientY: e.clientY }, () => {
				handleMouseUp(e);
			});
		};

		const handleMouseUp = (e: MouseEvent) => {
			if (!isDragging.value) return;
			isDragging.value = false;
			if (!verifySuccess.value) {
				currentX.value = 0;
			}
		};
		//移动历览器
		const isTouchStarted = ref(false);
		const touchStartHandler = (event: TouchEvent) => {
			// 阻止默认行为，例如防止页面滚动
			event.preventDefault();
			if (!isTouchStarted.value) {
				isTouchStarted.value = true;
				// 在这里添加触摸开始时的逻辑
				//console.log('触摸开始，执行相关操作', event);
				onstart(event.touches[0].clientX);
			} else {
				// 如果已经触摸开始，停止后续操作
				//console.log('已经触摸开始，停止本次操作');
			}
		};

		const touchMoveHandler = (e: TouchEvent) => {
			if (isTouchStarted.value) {
				onMove(
					{ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY },
					() => {
						isTouchStarted.value = false;
					}
				);
				// 在这里添加触摸移动时的逻辑
				//console.log('触摸移动，执行相关操作', e);
			} else {
				//console.log('尚未触摸开始，不执行移动操作');
			}
		};

		onMounted(() => {
			isMobile.value = isMobileBrowser();
			if (!isMobile.value) {
				document.addEventListener('mousemove', handleMouseMove);
				document.addEventListener('mouseup', handleMouseUp);
			}

			bgWidth.value = BgRef.value?.clientWidth || 0;
		});

		onUnmounted(() => {
			if (!isMobile.value) {
				document.removeEventListener('mousemove', handleMouseMove);
				document.removeEventListener('mouseup', handleMouseUp);
			}
		});

		return () => (
			<>
				<div
					ref={BgRef}
					class={style.dragVerify}
					style={{
						width: `${props.width}px`,
						height: `${props.height}px`,
					}}>
					{/* 已拖动进度背景 */}
					<div
						class="draged"
						style={{
							width: `${currentX.value}px`,
						}}
					/>
					{/* 滑块 */}
					<ElButton
						ref={sliderRef}
						class="slider"
						icon={ArrowRight}
						style={{
							left: `${currentX.value}px`,
						}}
						{...(isMobile.value
							? {
									onTouchstart: touchStartHandler,
									onTouchmove: touchMoveHandler,
							  }
							: {
									onMousedown: handleMouseDown,
							  })}
					/>

					{/* 提示信息 */}
					<div class={{ tip: true, success: verifySuccess.value }}>
						{verifySuccess.value
							? props.successText || message.value
							: props.text}
					</div>
				</div>
			</>
		);
	},
});

export default DragVerify;
