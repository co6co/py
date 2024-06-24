import { ref, defineComponent } from 'vue'
import { IEnumSelect } from 'co6co'
import { ElSelect, ElOption } from 'element-plus'

export default defineComponent({
  name: 'EnumSelect',
  props: {
    data: {
      type: Array<IEnumSelect>,
      required: true
    },
    modelValue: {
      type: [String, Number]
    },
    placeholder: {
      type: String,
      default: '请选择'
    }
  },
  emits: {
    'update:modelValue': (v: any) => true
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
          placeholder={prop.placeholder}
        >
          {prop.data.map((d, index) => {
            return <ElOption key={d.key} label={d.label} value={d.value}></ElOption>
          })}
        </ElSelect>
      )
    }
  }
})
