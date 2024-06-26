import { defineComponent, ref, reactive, type PropType, provide, inject } from 'vue'
import type { InjectionKey } from 'vue'

import { DialogForm, FormOperation, showLoading, closeLoading } from 'co6co'
import type { ObjectType, FormData, IResponse } from 'co6co'

import * as api from '../../api/group'
import * as types from 'co6co'

import {
  ElButton,
  ElFormItem,
  ElInputNumber,
  ElInput,
  ElMessage,
  type FormRules
} from 'element-plus'

export interface Item {
  id: number
  name: string
  priority: number
}
//Omit、Pick、Partial、Required
export type FormItem = Omit<Item, 'id'>

export default defineComponent({
  name: 'EditPriority',
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 90
    }
  },
  emits: {
    saved: (response: any, submitData: FormItem) => true
  },
  setup(prop, ctx) {
    const diaglogForm = ref<InstanceType<typeof DialogForm>>()

    const data = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        name: '',
        priority: 99999
      }
    })
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', data.fromData)

    provide('title', prop.title)

    const init_data = (oper: FormOperation, item?: Item) => {
      data.operation = oper
      switch (oper) {
        case FormOperation.add:
          data.id = -1
          data.fromData.name = ''
          data.fromData.priority = 99999

          break
        case FormOperation.edit:
          if (!item) return false
          data.id = item.id
          data.fromData.name = item.name
          data.fromData.priority = item.priority
          break
      }
      return true
    }

    const rules: FormRules = {
      priority: [{ required: true, message: '请输入优先级', trigger: 'blur' }]
    }

    const save = () => {
      //提交数据
      let promist: Promise<IResponse<any>>
      switch (data.operation) {
        case FormOperation.add:
          promist = api.set_boat_priority(0, data.fromData)
          break
        case FormOperation.edit:
          promist = api.set_boat_priority(data.id, data.fromData)
          break
        default:
          return
      }
      showLoading()
      promist
        .then((res) => {
          if (res.code == 0) {
            diaglogForm.value?.closeDialog()
            ElMessage.success(res.message)
            ctx.emit('saved', res.data, data.fromData)
          } else {
            ElMessage.error(`操作失败:${res.message}`)
          }
        })
        .finally(() => {
          closeLoading()
        })
    }

    const fromSlots = {
      buttons: () => (
        <>
          <ElButton
            onClick={() => {
              diaglogForm.value?.validate(save)
            }}
          >
            保存
          </ElButton>
        </>
      ),
      default: () => (
        <>
          <ElFormItem label="名称" prop="name" style="with:80px">
            <ElInput readonly v-model={data.fromData.name}></ElInput>
          </ElFormItem>

          <ElFormItem label="优先级" prop="priority">
            <ElInputNumber min={0} max={99999} v-model={data.fromData.priority}></ElInputNumber>
          </ElFormItem>
        </>
      )
    }
    const openDialog = (item: Item) => {
      init_data(FormOperation.edit, item)
      diaglogForm.value?.openDialog()
    }

    const rander = (): ObjectType => {
      return (
        <DialogForm
          title={prop.title}
          labelWidth={prop.labelWidth}
          style={ctx.attrs}
          rules={rules}
          ref={diaglogForm}
          v-slots={fromSlots}
        ></DialogForm>
      )
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
