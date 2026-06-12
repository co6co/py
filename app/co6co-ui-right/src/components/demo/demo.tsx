import { defineComponent, ref, onMounted } from 'vue'
//import {  getCurrentInstance} from 'vue'
import { ElInput } from 'element-plus'

export default defineComponent({
    components: { ElInput },
    // 直接根节点 TSX render
    setup(_, { attrs, slots  }) {
        // 原有逻辑不变...
        //const attrs = useAttrs()
        //const slots = useSlots()
        const refEle = ref<InstanceType<typeof ElInput>>()
        //const instance = getCurrentInstance()
        onMounted(() => { /* ... */ }) 
        return { attrs, slots, refEle }
    }, 
    // 直接根节点 TSX render
    render() {
        const { attrs, slots, refEle } = this 
        const scopedSlots = Object.fromEntries(
            Object.entries(slots).map(([slotName, slotFn]) => {
                // 作用域插槽：接收 scopedata 并透传给原插槽
                return [ slotName, (scopedata: Record<string, any>) =>    slotFn ? slotFn(scopedata) : null]
            })
        )
        return (
            <div>
                <ElInput ref={refEle} v-bind={attrs} v-slot={scopedSlots}>

                </ElInput>
            </div>
        )
    }
})