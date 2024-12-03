import { defineComponent, compile } from 'vue'

// 模板字符串
const templateString = `<div>
  <h1>{{ title }}</h1>
  <p>{{ content }}</p>
</div>`

// 编译模板字符串为渲染函数
const renderFunction = compile(templateString)

// 创建组件定义
export default defineComponent({
  name: 'DynamicComponent',
  props: {
    title: String,
    content: String
  },
  // 使用编译后的渲染函数
  render: renderFunction
})
