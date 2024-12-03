import { defineComponent, h, compile, ref } from 'vue'
export default defineComponent({
  setup() {
    // 定义一个模板字符串
    const template = '<div><h1>Hello {{ name }} </h1></div>'

    // 编译模板字符串为渲染函数
    const renderFunction = compile(template)
    const name = ref('小小')
    return { name, renderFunction }
  },
  template: '<div><h1>Hello {{ name }} </h1></div>'
})
