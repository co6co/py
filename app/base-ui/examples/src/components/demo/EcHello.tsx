import { defineComponent } from 'vue'
import type { PropType } from 'vue'
export default defineComponent({
  props: {
    count: { type: Number, required: true },
    person: {
      type: Object as PropType<{ name: string }>
    },
    color: {
      type: String as PropType<'success' | 'error' | 'primary'>,
      required: true
    }
  },
  setup(props, ctx) {
    return () => <div class={props.color}>{props.person?.name}</div>
  }
})
