import { defineComponent, ref, reactive, type PropType, provide, inject } from 'vue'
import type { InjectionKey } from 'vue'
import { default as EcDiaglogForm } from '../../common/EcDiaglogForm'

import { FormOperation } from '../../common/types'
import type { ObjectType, FormData } from '../../common/types'

import * as api from '../../../api/site'
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
import {  showLoading, closeLoading } from '@/components/Logining'


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
export type FormItem = Omit<Item, 'id' | 'createTime' | 'updateTime'>& { configs: Partial<api.ConfigItem>[] } &{removeConfig:number[]}
 
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

    const data = reactive<FormData<number, FormItem >>({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        name: '',
        postionInfo: '',
        deviceCode: '',
        deviceDesc: '',
        configs: [],
        removeConfig:[]
      }
    })
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', data.fromData) 

    const init_data = (oper: FormOperation, item?: Item) => {
      data.operation = oper
      data.fromData.configs = []
      data.fromData.removeConfig=[]
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

          if (item) {
            api.get_config_svc(item.id, api.ConfigCategory.devConfig).then((res) => {
              data.fromData.configs = res.data
            })
          }
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
      let promist: Promise<IResponse >
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
      showLoading()
      promist.then((res) => {
        if (res.code == 0) {
          diaglogForm.value?.closeDialog()
          ElMessage.success(`操作成功`)
          ctx.emit('saved', res.data)
        } else {
          ElMessage.error(`操作失败:${res.message}`)
        }
      })
      .finally(() => {
        closeLoading()
      })
    }

    const removeConfig = (index: number,item:Partial<api.ConfigItem>) => {
      if (item.id){
        data.fromData.removeConfig.push(item.id)
      }
      data.fromData.configs.splice(index, 1)
    }
    const addConfig = () => {
      data.fromData.configs.push({ name: '', category: api.ConfigCategory.devConfig, value: '' })
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
          {data.fromData.configs.map((item, index) => {
            return (
              <ElRow>
                <ElCol span={6}>
                  <ElFormItem
                    label={`配置${index + 1}名称`}
                    prop={'configs.' + index + '.name'}
                    rules={{
                      required: true,
                      message: `配置${index + 1}名称`,
                      trigger: 'blur'
                    }}
                  >
                    <ElInput v-model={item.name} placeholder="配置名称"></ElInput>
                  </ElFormItem>
                </ElCol>
                <ElCol span={16}>
                  <ElFormItem
                    label={`配置${index}内容`}
                    prop={'configs.' + index + '.value'}
                    rules={{
                      required: true,
                      message: `配置${index + 1}内容`,
                      trigger: 'blur'
                    }}
                  >
                    <ElInput v-model={item.value} placeholder="配置内容"></ElInput>
                  </ElFormItem>
                </ElCol>
                <ElCol span={2}>
                  <ElButton
                    onClick={() => {
                      removeConfig(index,item)
                    }}
                    icon={Minus}
                  ></ElButton>
                </ElCol>
              </ElRow>
            )
          })}
          <ElRow>
            <ElCol span={22}></ElCol>
            <ElCol span={2}>
              <ElButton onClick={addConfig} icon={Plus}></ElButton>
            </ElCol>
          </ElRow>
        </>
      )
    }
    const openDialog = (oper: FormOperation, item?: Item) => {
      init_data(oper, item) 
      diaglogForm.value?.openDialog( )
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
