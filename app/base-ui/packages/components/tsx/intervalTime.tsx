import { defineComponent, onUnmounted, ref } from 'vue'
//https://zhuanlan.zhihu.com/p/500610564
/**
const Interval = 5000 //30s
let _timer: NodeJS.Timeout | null = null
let isRunning = false
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const IntervalTime = function <T>(promise?: Promise<T>, bck?: (data: T) => void) {
  if (_timer) clearInterval(_timer)
  if (isRunning) return
  isRunning = true
  if (promise) {
    promise
      .then((res) => {
        if (bck) bck(res)
      })
      .catch((e) => {
        console.error(e)
      })
      .finally(() => {
        _timer = setInterval(IntervalTime, Interval, promise, bck)
        isRunning = false
      })
  }
}
onUnmounted(() => {
  if (_timer) clearInterval(_timer)
}) 
export default IntervalTime
 */
export default defineComponent({
  name: 'IntervalTime',
  props: {
    interval: {
      type: Number,
      default: 5000,
    },
  },
  setup(props, context) {
    const isRunning = ref(false)
    let timer: any = null //NodeJS.Timeout | null = null
    const runinng = function <T>(
      exec?: () => Promise<T>,
      bck?: (data: T) => void
    ) {
      if (timer) clearInterval(timer)
      if (isRunning.value) return
      isRunning.value = true
      if (exec) {
        exec()
          .then((res) => {
            if (bck) bck(res)
          })
          .catch((e) => {
            //console.error(e.message)
          })
          .finally(() => {
            timer = setInterval(runinng, props.interval, exec, bck)
            isRunning.value = false
          })
      }
    }
    // onMounted(() => {})
    onUnmounted(() => {
      if (timer) clearInterval(timer)
    })

    context.expose({
      runinng,
    })
    //return () => {}
  },
})
