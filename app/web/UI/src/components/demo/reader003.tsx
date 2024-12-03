import { defineComponent, h } from 'vue'

export default defineComponent({
  name: 'Reader_003',
  setup() {
    const message = 'Hello, Vue 3!'
    const reversedMessage = message.split('').reverse().join('')

    return () =>
      h('div', [
        h('p', message),
        h('p', reversedMessage),
        h('button', { onClick: () => alert(message) }, 'Greet')
      ])
  }
})
