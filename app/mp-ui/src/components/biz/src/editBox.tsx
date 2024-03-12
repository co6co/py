import { defineComponent, ref, reactive, type PropType, provide } from 'vue'
import { default as EcDiaglogForm } from '../../common/EcDiaglogForm'
import type { ObjectType, FormData } from '../../common/types'
import { FormOperation } from '../../common/types'
import * as api from '../../../api/site'
import {
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules,
  type FormInstance
} from 'element-plus'

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
  name: 'diaglogForm',
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 150
    }
  },
  emits: {
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const diaglogForm = ref<InstanceType<typeof EcDiaglogForm>>()
    const data = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        name: '',
        postionInfo: '',
        deviceCode: '',
        deviceDesc: ''
      }
    })

    provide('formData', data.fromData)
    console.info("ddddd",prop.title)
    provide('title', prop.title)
    const init_data = (oper: FormOperation, item?: Item) => {
      data.operation = oper
      switch (oper) {
        case FormOperation.add:
          data.id = -1
          data.fromData.name = ''
          data.fromData.postionInfo = ''
          data.fromData.deviceCode = ''
          data.fromData.deviceDesc = ''
          break
        case FormOperation.edit:
          if (!item) return false
          data.id = item.id
          data.fromData.name = item.name
          data.fromData.postionInfo = item.postionInfo
          data.fromData.deviceCode = item.deviceCode
          data.fromData.deviceDesc = item.deviceDesc
          break
      }
      return true
    }
    const rules: FormRules = {
      name: [{ required: true, message: '请输入设备名称', trigger: 'blur' }],
      postionInfo: [{ required: true, message: '请输入位置信息', trigger: 'blur' }],
      deviceCode: [{ required: true, message: '请输入设备代码', trigger: 'blur' }],
      deviceDesc: [{ required: true, message: '请输入设备用途', trigger: 'blur' }]
    }

    const save = () => {
      //提交数据
      let promist: Promise<IPageResponse<any>>
      switch (data.operation) {
        case FormOperation.add:
          promist = api.add_site_svc(data.fromData)
          break
        case FormOperation.edit:
          promist = api.edit_site_svc(data.id, data.fromData)
          break
        default:
          return
      }
      promist.then((res) => {
        if (res.code == 0) {
          diaglogForm.value?.closeDialog()
          ElMessage.success(`操作成功`)
          ctx.emit('saved', res.data)
        } else {
          ElMessage.error(`操作失败:${res.message}`)
        }
      })
    }

    const fromSlots = {
      buttons: () => (
        <>
          <ElButton
            onClick={() => {
              diaglogForm.value?.formInstance?.save()
            }}
          >
            保存
          </ElButton>
        </>
      ),
      default: () => (
        <>
          <ElFormItem label="名称" prop="name">
            <ElInput v-model={data.fromData.name}></ElInput>
          </ElFormItem>
          <ElFormItem label="设备代码" prop="deviceCode">
            <ElInput v-model={data.fromData.deviceCode} placeholder="设备代码"></ElInput>
          </ElFormItem>
          <ElFormItem label="用途" prop="deviceDesc">
            <ElInput
              v-model={data.fromData.deviceDesc}
              type="textarea"
              placeholder="装用途，抓拍描述等"
            ></ElInput>
          </ElFormItem>
          <ElFormItem label="设备位置" prop="postionInfo">
            <ElInput
              v-model={data.fromData.postionInfo}
              type="textarea"
              placeholder="位置信息"
            ></ElInput>
          </ElFormItem>
        </>
      )
    }
    const openDialog = (oper: FormOperation, item?: any) => {
      init_data(oper, item)
      diaglogForm.value?.openDialog(oper, item)
    }

    const rander = (): ObjectType => {
      return (
        <EcDiaglogForm
          style={ctx.attrs}
          rules={rules}
          ref={diaglogForm}
          v-slots={fromSlots}
          onSubmit={() => {
            save()
          }}
        ></EcDiaglogForm>
      )
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
