import { ref, reactive, defineComponent, type PropType } from 'vue'

import {
  ElForm,
  ElFormItem,
  ElSelect,
  ElOption,
  ElSwitch,
  ElInputNumber,
  ElMessage,
  ElButton,
  ElRadioGroup,
  ElRadio,
  type FormInstance,
  type FormRules,
  ElInput
} from 'element-plus'
export interface formDataType {
  visible: boolean
  title?: string
  loading: boolean
}
export default defineComponent({
  name: 'EcForm',
  props: {
    rules:{
      type: Object as PropType<FormRules>,
      required:false
    },
    model: {
      type: Object as PropType<ObjectConstructor>,
      required: true
    }
  },
  emits: {
    close: () => true
  },
  setup(prop, context) {
    const formRef = ref<FormInstance>()
    const formData = reactive<formDataType>({
      visible: false,
      loading: false
    })
    //其他api 操作
    //end
    const onOpenDialog = (title: string) => {
      formData.visible = true
      formData.title = title
    }
    const save = (formEl: FormInstance | undefined) => {
      if (!formEl) return

      formEl.validate((value) => {
        if (value) {
          /**
             formData.loading = true
             api
            .set_svc({ data: to_parm()  })
            .then((res) => {
                if (pd_api.isSuccess(res)) {
                ElMessage.success(`编辑成功`)
                getData()
                } else {
                ElMessage.error(`编辑失败:${res.message}`)
                }
            })
            .finally(() => {
                form.loading = false
            })
             */
        } else {
          ElMessage.error('请检查输入的数据！')
          return false
        }
      })
    }
    context.expose({
      onOpenDialog
    })
    const dialogSlots = {
      footer: () => {
        return (
          <span class="dialog-footer">
            <ElButton
              onClick={() => {
                formData.visible = false
                context.emit('close')
              }}
            >
              关闭
            </ElButton>
            <slot name="buttons"></slot>
          </span>
        )
      }
    }
    return () => {
      //可以写某些代码
      return (
        <>
          <ElForm labelWidth={150} ref={formRef} rules={prop.rules}  model={prop.model}></ElForm>
        </>
      )
    }
  }
})
