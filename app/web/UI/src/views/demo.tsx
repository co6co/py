import { defineComponent, ref, VNode } from 'vue'

import { DictSelect } from 'co6co-right'
export default defineComponent({
  setup(prop, ctx) {
    //存储本地值
    const localValue = ref(0)

    const rander = (): VNode => {
      return (
        <>
          <span>{localValue.value}</span>
          <DictSelect
            dictTypeCode="SYS_TASK_CATEGORY"
            v-model={localValue.value}
            placeholder="请选择类型"
          ></DictSelect>
        </>
      )
    }
    return rander
  }
})
