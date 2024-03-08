import { ref, reactive, onUnmounted, defineComponent, watchEffect ,defineExpose} from 'vue';
 

export const InputNumber=defineComponent({
    name:"InputNumber",
    setup(props){
        return ()=>(
            <div class="demoInput"></div>
        )
    }
}) 