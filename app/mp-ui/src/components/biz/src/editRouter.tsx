import { defineComponent, ref, reactive, type PropType, provide, inject } from 'vue'
import type { InjectionKey } from 'vue'
import { default as EcDiaglogForm } from '../../common/EcDiaglogForm'

import { FormOperation } from '../../common/types'
import type { ObjectType, FormData } from '../../common/types'

import * as api from '../../../api/site/router'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules,
  type FormInstance
} from 'element-plus'
import { Plus, Minus } from '@element-plus/icons-vue'

//Omit、Pick、Partial、Required
export type FormItem = Omit<api.Item, 'id' | 'createTime' | 'updateTime'>

export default defineComponent({
  name: 'diaglogForm',
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
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const diaglogForm = ref<InstanceType<typeof EcDiaglogForm>>()

    const data = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        name: '',
        siteId: 0,
        uuid: '',
        innerIp: '',
        ip: '',
        sim: '',
        ssd: '',
        password: ''
      }
    })
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', data.fromData)
    provide('title', prop.title)

    const init_data = (oper: FormOperation, siteId: number, item?: api.Item) => {
      data.operation = oper
      switch (oper) {
        case FormOperation.add:
          data.id = -1
          data.fromData.name = ''
          data.fromData.siteId = siteId
          data.fromData.uuid = ''
          data.fromData.innerIp = ''
          data.fromData.ip = ''
          data.fromData.sim = ''
          data.fromData.ssd = ''
          data.fromData.password = ''
          break
        case FormOperation.edit:
          if (!item) return false
          data.id = item.id
          data.fromData.name = item.name
          data.fromData.siteId = siteId
          data.fromData.uuid = item.uuid
          data.fromData.innerIp = item.innerIp
          data.fromData.ip = item.ip
          data.fromData.sim = item.sim
          data.fromData.ssd = item.ssd
          data.fromData.password = item.password
          break
      }
      return true
    }
    const rules: FormRules = {
      name: [{ required: true, message: '请输入设备名称', trigger: 'blur' }],
      innerIp: [{ required: true, message: '请输入位置信息', trigger: 'blur' }],
      ip: [{ required: true, message: '请输入设备代码', trigger: 'blur' }],
      password: [{ required: true, message: '请输入设备代码', trigger: 'blur' }],
      sim: [{ required: true, message: '请输入设备用途', trigger: 'blur' }],
      ssd: [{ required: true, message: '请输入设备代码', trigger: 'blur' }]
    }

    const save = () => {
      //提交数据
      let promist: Promise<IPageResponse<any>>
      switch (data.operation) {
        case FormOperation.add:
          promist = api.add_svc(data.fromData)
          break
        case FormOperation.edit:
          promist = api.edit_svc(data.id, data.fromData)
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
              diaglogForm.value?.validate(save)
            }}
          >
            保存
          </ElButton>
        </>
      ),
      default: () => (
        <>
          <ElFormItem label="设备UUID" prop="uuid">
            <ElInput v-model={data.fromData.uuid}></ElInput>
          </ElFormItem>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="名称" prop="name">
                <ElInput v-model={data.fromData.name}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="设备密码" prop="password">
                <ElInput v-model={data.fromData.password}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="内网IP地址" prop="innerIp">
                <ElInput v-model={data.fromData.innerIp}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="IP地址" prop="ip">
                <ElInput v-model={data.fromData.ip}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="sim号" prop="sim">
                <ElInput v-model={data.fromData.sim}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="WIFI ssd" prop="ssd">
                <ElInput v-model={data.fromData.ssd}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>
        </>
      )
    }
    const openDialog = (siteId: number) => {
      api.get(siteId).then((res) => {
        let oper = FormOperation.add
        if (res.data) {
          oper = FormOperation.edit
        }
        init_data(oper, siteId, res.data)
        diaglogForm.value?.openDialog()
      })
    }

    const rander = (): ObjectType => {
      return (
        <EcDiaglogForm
          title={prop.title}
          labelWidth={prop.labelWidth}
          style={ctx.attrs}
          rules={rules}
          ref={diaglogForm}
          v-slots={fromSlots}
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
