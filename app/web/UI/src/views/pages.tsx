import { defineComponent, onMounted, ref, VNode, compile } from 'vue'
import { component as api } from '../api/biz'
export default defineComponent({
  async setup(prop, ctx) {
    const resource_code = ref('')
    const res = await api.get_component_code('page01')
    resource_code.value = res.data

    // 解析组件代码
    const componentDefinition = compile(resource_code.value)
    console.info(componentDefinition)
    /*
    const rander = (): VNode => {
      return <div style="padding:5px"></div>
    }
    return rander
    */
    return componentDefinition
  }
})
