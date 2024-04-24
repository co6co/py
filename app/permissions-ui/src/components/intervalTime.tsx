import { onUnmounted, defineComponent, ref , onMounted} from 'vue' 

export default defineComponent({
  name: 'IntervalTime',
  props: {
    interval: {
      type: Number,
      default: 5000
    }
  },
  setup(props, context) {
    const isRunning = ref(false)
    let timer: any=null//NodeJS.Timeout | null = null
    const runinng = function <T>(exec?: ()=>  Promise<T>, bck?: (data: T) => void) {
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
    onMounted(()=>{ 
    })
    onUnmounted(()=>{
      

      if(timer) clearInterval(timer) 
    }) 

    context.expose({
      runinng
    });  
    return () => {} 
  }
}) 