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
import type { ObjectType } from './types'
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
      required: false
    },
    model: {
      type: Object, //as PropType<ObjectConstructor>,
      required: true
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 150
    }
  },
  emits:{
    error:(msg:string)=>true,
    submit:( )=>true
  },
  setup(prop, {attrs,slots,emit,expose}) {
    const formRef = ref<FormInstance>() 

    const save = (instance: FormInstance | undefined) => {
      if (!instance) {
        ElMessage.error('表单对象为空！')
        emit("error","表单对象为空！") 
        return
      }
      console.info("formData",prop.model)
      instance.validate((value) => {
        if (!value) {
          ElMessage.error('请检查输入的数据！') 
          emit("error","请检查输入的数据！") 
          return false
        }
        //提交数据 
        emit("submit")  
      })
    } 
    const onSave=()=>{
      save(formRef.value)
    }
    const render = (): ObjectType => {
      //可以写某些代码
      return (
        <ElForm labelWidth={prop.labelWidth} ref={formRef} rules={prop.rules} model={prop.model}>
          {slots.default ? slots.default():null}
        </ElForm>
      )
    }
    expose({
      formInstance:formRef ,
      save:onSave  
    }) 
    render.formInstance = formRef
    render.save=onSave
    return render
  }
})
