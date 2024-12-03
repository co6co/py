import { defineComponent, compile } from 'vue'

export default defineComponent({
  name: 'Basis_001',
  data() {
    return {
      message: 'Hello, Vue 3!'
    }
  },
  methods: {
    greet() {
      alert(this.message)
    }
  },
  computed: {
    reversedMessage() {
      return this.message.split('').reverse().join('')
    }
  },
  watch: {
    message(newValue, oldValue) {
      console.log('message changed from', oldValue, 'to', newValue)
    }
  },
  template: `<div>
    <p>{{ message }}</p>
    <p>{{ reversedMessage }}</p>
    <button @click="greet">Greet</button>
  </div>`
})
