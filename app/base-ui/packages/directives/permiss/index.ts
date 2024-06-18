import { usePermissStore } from './hook'
import type { DirectiveBinding, ObjectDirective } from 'vue'

const permissDirective: ObjectDirective = {
  mounted(el: HTMLElement, binding: DirectiveBinding) {
    const store = usePermissStore()
    if (!store.includes(String(binding.value))) {
      el['hidden'] = true
    }
  },
}
export default permissDirective
export { usePermissStore }
