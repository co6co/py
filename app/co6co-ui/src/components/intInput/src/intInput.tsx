import { defineComponent } from 'vue'
import { ElInput } from 'element-plus'

export default defineComponent({
  name: 'IntInput',
  props: {
    modelValue: {
      type: Number,
      required: true
    }
  },
  emits: ['update:modelValue'],
  setup(props, { emit ,attrs}) {
    const handleInput = (val: string) => {
      const num = parseFloat(val)
      emit('update:modelValue', isNaN(num) ? 0 : num)
    }

    return () => (
      <ElInput
        {...attrs} 
        modelValue={String(props.modelValue)}
        onInput={handleInput}
        {...{ onChange: handleInput }} // 防止中文输入法问题
      />
    )
  }
})