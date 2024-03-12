import { defineComponent, ref, reactive, type PropType, computed, inject } from 'vue'
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
          <ElButton onClick={() => formInstance.value?.save()}>保存</ElButton>
        </>
      )
    }

    const data: Object = inject('formData') || {}
    //console.warn("data",data)
    const openDialog = (oper: FormOperation, item?: any) => {
      if (dialogRef.value) {
        dialogRef.value.data.title = prop.title||"弹出框"
        setDiaglogVisible(true)
      }
    }
    const closeDialog = () => {
      setDiaglogVisible(false)
    }
    ctx.expose({
      openDialog,
      closeDialog,
      formInstance
    })
    const rander = (): ObjectType => {
      return (
        <EcDialog
          style={ctx.attrs}
          ref={dialogRef}
          v-slots={ctx.slots.buttons ? { buttons: ctx.slots.buttons } : slots}
        >
          <EcForm
            labelWidth={prop.labelWidth}
            v-slots={ctx.slots.default}
            ref={formInstance}
            rules={prop.rules}
            model={data}
            onSubmit={() => { 
              ctx.emit('submit')
            }}
          ></EcForm>
        </EcDialog>
      )
    }
    rander.openDialog = openDialog
    rander.closeDialog = closeDialog
    rander.formInstance = formInstance
    return rander
  } //end setup
})
