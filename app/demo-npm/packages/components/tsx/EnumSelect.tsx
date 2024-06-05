import { defineComponent, ref, type PropType } from 'vue'
import { ElOption, ElSelect } from 'element-plus'
import type { IEnumSelect } from '@co6co/constants'

export default defineComponent({
  name: 'EnumSelect',
  props: {
    data: {
      type: Object as PropType<Array<IEnumSelect>>,
      required: true,
    },
    modelValue: {
      type: [String, Number],
    },
    placeholder: {
      type: String,
      default: '请选择',
    },
  },
  emits: {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    'update:modelValue': (v: any) => true,
  },
  setup(prop, context) {
    const DATA = ref<undefined | string | number>(prop.modelValue)
    const onChanged = () => {
      context.emit('update:modelValue', DATA.value)
    }
    return () => {
      //可以写某些代码
      return (
        <ElSelect
          clearable
          style={context.attrs}
          class="mr10"
          v-model={DATA.value}
          onChange={onChanged}
          placeholder="请选择"
        >
          {prop.data.map((d) => {
            return (
              <ElOption key={d.key} label={d.label} value={d.value}></ElOption>
            )
          })}
        </ElSelect>
      )
    }
  },
})
