<template>
  <div>
    <p>{{ message }}</p>
    <p>{{ reversedMessage }}</p>
    <button @click="greet">Greet</button>
  </div>
</template>

<script>
import { defineComponent, ref, computed } from 'vue'

export default defineComponent({
  name: 'SignalFile004',
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
  }
})
</script>

<style scoped>
/* 组件的样式 */
</style>
