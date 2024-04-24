import { defineComponent, ref, reactive,  computed, inject } from 'vue'
import type{InjectionKey,PropType} from 'vue'
import EcDialog, { type dialogDataType } from './EcDialog'
import { default as EcForm } from './EcForm'
import type { ObjectType, FormData } from './types'
import { FormOperation } from './types' 

import {
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules,
  type FormInstance
} from 'element-plus'

//Omit、Pick、Partial、Required
export interface Item {
  id: number
  name: string
  postionInfo: string
  deviceCode: string
  deviceDesc: string
  createTime: string
  updateTime: string
}
//Omit、Pick、Partial、Required 

export type FormItem = Omit<Item, 'id' | 'createTime' | 'updateTime'>
export default defineComponent({
  name: 'EcdiaglogForm',
  props: {
    title: {
      type: String
    },
    rules: {
      type: Object as PropType<FormRules>
    },
    labelWidth: {
        type: Number, //as PropType<ObjectConstructor>,
        default: 150
    }
  },
  emits: {
    submit: () => true
  },
  setup(prop, ctx) {
    const dialogRef = ref<InstanceType<typeof EcDialog>>()
    const formInstance = ref<InstanceType<typeof EcForm>>()
    const setDiaglogVisible = (show: boolean) => {
      if (dialogRef.value) {
        dialogRef.value.data.visible = show
      }
    }
    const slots = {
      buttons: () => (
        <>
          <ElButton onClick={() => formInstance.value?.validate()}>保存</ElButton>
        </>
      )
    } 
   // const data: Object = inject('formData') || {} 
  const data: Object = inject('formData') || {} 
  const openDialog = (  ) => { 
      if (dialogRef.value) {
        dialogRef.value.data.title = prop.title||"弹出框"
        setDiaglogVisible(true)
      }
    }
    const closeDialog = () => {
      setDiaglogVisible(false)
    }
   
    const rander = (): ObjectType => { 
      return (
        <EcDialog title={prop.title}
          style={ctx.attrs}
          ref={dialogRef}
          v-slots={ctx.slots.buttons ? { buttons: ctx.slots.buttons } : slots}
        >
          <EcForm  style="0 45px 0 0"
            labelWidth={prop.labelWidth}
            v-slots={ctx.slots.default}
            ref={formInstance}
            rules={prop.rules}
            model={data} 
          ></EcForm>
        </EcDialog>
      )
    }
    //暴露方法给父组件

    const validate=(success:()=>void)=>{
      formInstance.value?.validate(success)
    }
    ctx.expose({
      openDialog,
      closeDialog,
      formInstance,
      validate
    })

    rander.openDialog = openDialog
    rander.closeDialog = closeDialog
    rander.formInstance = formInstance
    rander.validate=validate
    return rander
  } //end setup
})
