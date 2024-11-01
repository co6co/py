import { defineComponent, onMounted, ref, VNode, h, compile } from 'vue'
import { showLoading, closeLoading } from 'co6co'
import { component as api } from '../api/biz'

async function createDynamicComponent2(componentData: { template: string; script: string }) {
  const { template, script } = componentData

  // 使用 eval 将字符串形式的脚本转换为实际的函数
  const scriptFunction = eval(`(${script})`)

  // 定义组件
  const DynamicComponent = defineComponent({
    ...scriptFunction,
    setup(props) {
      const renderFunction = compile(template)
      return () => renderFunction() //({ ...props })
    }
  })

  return DynamicComponent
}
function createDynamicComponent(script: string) {
  // 使用 eval 将字符串形式的脚本转换为实际的函数
  console.info(script)
  const scriptFunction = eval(`(${script})`)
  const scriptFunction2 = new Function(script)
  console.info('122123', scriptFunction)
  return scriptFunction

  // 定义组件
  const DynamicComponent = defineComponent({
    ...scriptFunction
  })

  return DynamicComponent
}

const useNameic = () => {
  showLoading()
  const resource_code = ref(
    `
    //import { defineComponent, onMounted, ref, VNode, compile } from 'vue'
    //import { component as api } from '../api/biz'
    export default defineComponent({
      setup(prop, ctx) { 
        const rander = (): VNode => {
          return <div style="padding:5px">123456</div>
        }
        return rander    
      }
    }) 
    `
  )

  //const res = await api.get_component_code('page01')
  //resource_code.value = res.data

  closeLoading()
  // 解析组件代码
  return createDynamicComponent(resource_code.value)
}
const component = useNameic()
export default component
/*
export default defineComponent({
  setup(prop, ctx) {
    const rander = (): VNode => {
      return <div style="padding:5px">123456</div>
    }
    return rander
  }
})
*/
