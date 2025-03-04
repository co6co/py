// DragVerify.tsx
import { ElButton } from 'element-plus'
import { defineComponent, ref, reactive, onMounted, onUnmounted, watch } from 'vue'
import style from '@/assets/css/dragVerify.module.less'
import { IDragVerifyData, dragVerify_Svc } from '@/api/verify'
const DragVerify = defineComponent({
  props: {
    width: {
      type: Number,
      default: 300
    },
    height: {
      type: Number,
      default: 50
    },
    successText: {
      type: String,
      default: '验证成功'
    },
    text: {
      type: String,
      default: '拖动滑块完成验证'
    },
    onVerifySuccess: {
      type: Function,
      required: true
    },
    modelValue: {
      type: String
    }
  },
  emits: {
    'update:modelValue': (data: String) => true
  },
  setup(props, ctx) {
    const sliderRef = ref<InstanceType<typeof ElButton> | null>(null)
    const BgRef = ref<HTMLDivElement | null>(null)
    const bgWidth = ref(0)
    const isDragging = ref(false)
    const offsetX = ref(0)
    const currentX = ref(0)
    const verifySuccess = ref(false)

    const DATA = reactive<IDragVerifyData>({
      start: 0,
      end: 0
    })
    //存储本地值
    const localValue = ref(props.modelValue)
    // 监听 modelValue 的变化 更新本地值
    watch(
      () => props.modelValue,
      (newValue) => {
        localValue.value = newValue
        //需要重新验证时
        if (!newValue) {
          currentX.value = 0
          verifySuccess.value = false
        }
      }
    )

    const onChange = (newValue: string) => {
      localValue.value = newValue
      ctx.emit('update:modelValue', newValue)
    }

    const handleMouseDown = (e: MouseEvent) => {
      if (verifySuccess.value) return
      isDragging.value = true
      offsetX.value = e.clientX
      DATA.start = new Date().getTime()
    }
    const handleMouseMove = (e: MouseEvent) => {
      // 取消上一次的requestAnimationFrame请求
      if (!isDragging.value || verifySuccess.value) return
      const diffX = e.clientX - offsetX.value
      currentX.value = Math.min(
        Math.max(0, diffX),
        props.width - (sliderRef.value?.$el.offsetWidth || 0)
      )
      if (currentX.value >= bgWidth.value - (sliderRef.value?.$el.offsetWidth || 0)) {
        DATA.end = new Date().getTime()
        verifySuccess.value = true
        //停止
        handleMouseUp(e)
        dragVerify_Svc(DATA)
          .then((res) => {
            onChange(res.data)
            props.onVerifySuccess(res.data)
          })
          .catch((err) => {
            verifySuccess.value = false
            currentX.value = 0
            //不是预期的错误打印日志
            if (!err.code) console.log(err)
          })
          .finally(() => {})
      }
    }

    const handleMouseUp = (e: MouseEvent) => {
      if (!isDragging.value) return
      isDragging.value = false
      if (!verifySuccess.value) {
        currentX.value = 0
      }
    }

    onMounted(() => {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      bgWidth.value = BgRef.value?.clientWidth || 0
    })

    onUnmounted(() => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    })

    return () => (
      <>
        <div
          ref={BgRef}
          class={style.dragVerify}
          style={{
            width: `${props.width}px`,
            height: `${props.height}px`
          }}
        >
          {/* 已拖动进度背景 */}
          <div
            class="draged"
            style={{
              width: `${currentX.value}px`
            }}
          />
          {/* 滑块 */}
          <ElButton
            ref={sliderRef}
            class="slider"
            style={{
              left: `${currentX.value}px`
            }}
            onMousedown={handleMouseDown}
          >
            →
          </ElButton>

          {/* 提示信息 */}
          <div class={{ tip: true, success: verifySuccess.value }}>
            {verifySuccess.value ? props.successText : props.text}
          </div>
        </div>
      </>
    )
  }
})

export default DragVerify
