import { defineComponent, ref, computed } from 'vue'

export default defineComponent({
  name: 'Composition_002',
  setup() {
    const message = ref('Hello, Vue 3!')

    const reversedMessage = computed(() => {
      return message.value.split('').reverse().join('')
    })

    function greet() {
      alert(message.value)
    }

    return {
      message,
      reversedMessage,
      greet
    }
  },
  template: `<div>
    <p>{{ message }}</p>
    <p>{{ reversedMessage }}</p>
    <button @click="greet">Greet</button>
  </div>`
})
