import { type PropType, defineComponent, ref } from 'vue'

import {
  ElForm,
  ElMessage,
  type FormInstance,
  type FormRules,
} from 'element-plus'
import { type ObjectType } from '@co6co/constants'

export interface formDataType {
  visible: boolean
  title?: string
  loading: boolean
}

export default defineComponent({
  name: 'EcForm',
  props: {
    rules: {
      type: Object as PropType<FormRules>,
      required: false,
    },
    model: {
      type: Object, //as PropType<ObjectConstructor>,
      required: true,
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 150,
    },
  },
  emits: {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    error: (msg: string) => true,
    submit: () => true,
  },
  setup(prop, { attrs, slots, emit, expose }) {
    const formRef = ref<FormInstance>()

    const _checkData = (
      instance: FormInstance | undefined,
      successBck?: () => void,
      validateBefore?: () => void
    ) => {
      if (!instance) {
        ElMessage.error('表单对象为空！')
        emit('error', '表单对象为空！')
        return false
      }
      if (validateBefore) validateBefore()
      instance.validate((isValid) => {
        if (!isValid) {
          ElMessage.error('请检查输入的数据！')
          emit('error', '请检查输入的数据！')
          /*
          return new Promise((resolve) => {
            resolve()
          })
          */
          return Promise.reject('valid Form Error')
          //return false;
        }
        //提交数据
        if (successBck) successBck()
        emit('submit')
      })
    }
    const validate = (successBck?: () => void, validateBefore?: () => void) => {
      _checkData(formRef.value, successBck, validateBefore)
    }
    const render = (): ObjectType => {
      //可以写某些代码
      return (
        <ElForm
          style={attrs}
          labelWidth={prop.labelWidth}
          ref={formRef}
          rules={prop.rules}
          model={prop.model}
        >
          {slots.default ? slots.default() : null}
        </ElForm>
      )
    }
    expose({
      formInstance: formRef,
      validate,
    })
    render.formInstance = formRef
    render.validate = validate
    return render
  },
})
