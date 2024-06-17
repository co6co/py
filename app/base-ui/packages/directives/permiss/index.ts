import { usePermissStore } from './hook'
import type { DirectiveBinding, ObjectDirective } from 'vue'
const store = usePermissStore()

const Permiss: ObjectDirective = {
  mounted(el: HTMLElement, binding: DirectiveBinding) {
    if (!store.includes(String(binding.value))) {
      el['hidden'] = true
    }
  },
}

export { Permiss }
