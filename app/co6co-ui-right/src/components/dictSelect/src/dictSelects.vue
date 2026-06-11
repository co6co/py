<template>
    <div>
        <el-input ref="refEle"  v-bind="attrs">
            <template v-for="(_,name) in slots" #[name]="scopedata">
                <slot :name="name" v-bind="scopedata"></slot>
            </template> 
        </el-input> 
    </div>
</template> 
<script setup lang="ts">
import { ref,onMounted, useAttrs,useSlots,getCurrentInstance  } from 'vue'
import { ElInput } from 'element-plus'
const attrs = useAttrs()
const slots = useSlots()
const refEle = ref<InstanceType<typeof ElInput>>()
const instance = getCurrentInstance() 
onMounted(() => {
    console.log('onMounted')
    for (const key in refEle.value) {
        if (instance) {
            instance[key] = attrs[key]
        } 
    }
}) 
</script>