/* eslint-disable no-console */
/// <reference types="vite/client" />

import { defineComponent, h, onMounted, ref, resolveComponent } from 'vue'
import { ElEmpty, ElIcon, ElOption, ElSelect } from 'element-plus'
import * as icon from '@element-plus/icons-vue'
//import iconStyle from '@co6co/theme/eciconselect.module.css'

export default defineComponent({
  name: 'EcIconSelect',
  props: {
    modelValue: {
      type: String,
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
    const DATA = ref<undefined | string>(prop.modelValue)
    const onChanged = () => {
      context.emit('update:modelValue', DATA.value)
    }
    onMounted(() => {
      console.info(DATA.value, prop.modelValue)
    })
    const vsolft = {
      default: () => {
        //ul  class={iconStyle.iconList}
        //class={iconStyle.icon_item}

        return (
          <ul>
            {Object.keys(icon).map((key, index) => {
              return (
                <ElOption key={index} value={key}>
                  <ElIcon>{h(resolveComponent(key))}</ElIcon>
                </ElOption>
              )
            })}
          </ul>
        )
      },

      prefix: () => {
        return DATA.value ? (
          <ElIcon color="#c6c6c6">{h(resolveComponent(DATA.value))}</ElIcon>
        ) : (
          <>空</>
        )
      },
      empty: () => {
        return <ElEmpty></ElEmpty>
      },
    }
    return () => {
      //可以写某些代码
      return (
        <ElSelect
          clearable
          filterable={true}
          style={context.attrs}
          class="iconList"
          v-model={DATA.value}
          onChange={onChanged}
          placeholder="请选择图标"
          v-slots={vsolft}
        ></ElSelect>
      )
    }
  },
})
