import { defineComponent, ref, reactive, type PropType, provide, inject } from 'vue'
import type { InjectionKey } from 'vue'
import { default as EcDiaglogForm } from '../../common/EcDiaglogForm'

import { FormOperation } from '../../common/types'
import type { ObjectType, FormData } from '../../common/types'

import * as api from '../../../api/site/aiBox'

import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInputNumber,
  ElInput,
  ElMessage,
  type FormRules,
  type FormInstance
} from 'element-plus'
import {  showLoading, closeLoading } from '@/components/Logining'

//Omit、Pick、Partial、Required
export type FormItem = Omit<api.Item, 'id' | 'createTime' | 'updateTime'>

export default defineComponent({
  name: 'EditAIBox',
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
        code: '',
        innerIp: '',
        ip: '',
        cpuNo: '',
        mac: '',
        license: '',
        talkbackNo: ''
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
          data.fromData.siteId = siteId
          data.fromData.name = ''
          data.fromData.uuid = ''
          data.fromData.code = ''
          data.fromData.innerIp = ''
          data.fromData.ip = ''
          data.fromData.cpuNo = ''
          data.fromData.mac = ''
          data.fromData.license = ''
          data.fromData.talkbackNo = ''
          break
        case FormOperation.edit:
          if (!item) return false
          data.id = item.id
          data.fromData.name = item.name
          data.fromData.name = item.name
          data.fromData.uuid = item.uuid
          data.fromData.code = item.code
          data.fromData.innerIp = item.innerIp
          data.fromData.ip = item.ip
          data.fromData.cpuNo = item.cpuNo
          data.fromData.mac = item.mac
          data.fromData.license = item.license
          data.fromData.talkbackNo = item.talkbackNo
          break
      }
      return true
    }

    const rules: FormRules = {
      name: [{ required: true, message: '请输入名称', trigger: 'blur' }],
      code: [{ required: true, message: '请输入设备编码', trigger: 'blur' }],
      innerIp: [{ required: true, message: '请输入设备内部IP', trigger: 'blur' }],
      ip: [{ required: true, message: '请输入设备IP地址', trigger: 'blur' }],
      mac: [{ required: true, message: '请输入设备MAC地址', trigger: 'blur' }],
      talkbackNo: [{ type: 'number', required: true, message: '请输入对讲号', trigger: 'blur' }]
    }

    const save = () => {
      //提交数据
      let promist: Promise<IResponse<any>>
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
      showLoading()
      promist.then((res) => {
        if (res.code == 0) {
          diaglogForm.value?.closeDialog()
          ElMessage.success(`操作成功`)
          ctx.emit('saved', res.data)
        } else {
          ElMessage.error(`操作失败:${res.message}`)
        }
      })  .finally(() => {
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
              <ElFormItem label="设备编码" prop="code">
                <ElInput v-model={data.fromData.code}></ElInput>
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
              <ElFormItem label="CPU序列号" prop="cpuNo">
                <ElInput v-model={data.fromData.cpuNo}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="设备MAC地址" prop="mac">
                <ElInput v-model={data.fromData.mac}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="对讲号" prop="talkbackNo">
                <ElInputNumber
                  v-model={data.fromData.talkbackNo}
                  placeholder="对讲号"
                ></ElInputNumber>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}></ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={16}>
              <ElFormItem label="授权码" prop="license">
                <ElInput v-model={data.fromData.license} type="textarea"></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}></ElCol>
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
