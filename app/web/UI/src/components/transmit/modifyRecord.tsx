import {
  defineComponent,
  ref,
  reactive,
  computed,
  provide,
  onMounted,
  onBeforeUnmount,
  VNode,
  nextTick
} from 'vue'
import type { InjectionKey } from 'vue'
import {
  DialogForm,
  FormOperation,
  showLoading,
  closeLoading,
  FormItemBase,
  IResponse,
  EnumSelect,
  EnumSelectInstance,
  type DialogFormInstance,
  type FormData
} from 'co6co'

import { DictSelect, DictSelectInstance } from 'co6co-right'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules,
  ElSwitch,
  ElInputNumber
} from 'element-plus'

import * as api from '@/api/transimit/cf'

export interface Item extends api.IParam, FormItemBase {}

//Omit、Pick、Partial、Required
export type FormItem = Omit<Item, 'id' | 'createUser' | 'updateUser' | 'createTime' | 'updateTime'>
export default defineComponent({
  name: 'record',
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 110
    }
  },
  emits: {
    //@ts-ignore
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const diaglogForm = ref<DialogFormInstance>()
    const DATA = reactive<FormData<number, FormItem> & { proxiable: boolean }>({
      operation: FormOperation.add,
      id: 0,
      proxiable: true,
      fromData: {
        name: '',
        record_id: '',
        type: 'A',
        content: '',
        ttl: 3600,
        proxied: false
      }
    })
    //const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    //provide('formData', DATA.fromData)
    const save = () => {
      //提交数据
      let promist: Promise<IResponse>
      switch (DATA.operation) {
        case FormOperation.add:
          promist = api.add_svc(DATA.fromData)
          break
        case FormOperation.edit:
          promist = api.edit_svc(DATA.fromData)
          break
        default:
          ElMessage.error('未知操作')
          return
      }
      showLoading()
      promist
        .then((res) => {
          diaglogForm.value?.closeDialog()
          ElMessage.success(res.message || `操作成功`)
          ctx.emit('saved', res)
        })
        .finally(() => {
          closeLoading()
        })
    }
    onMounted(() => {})
    onBeforeUnmount(() => {})

    const rules: FormRules = {
      name: [{ required: true, message: '请输入子域名', trigger: ['blur', 'change'] }],
      type: [{ required: true, message: '请选择记录类型', trigger: ['blur', 'change'] }],
      content: [{ required: true, message: '记录值', trigger: ['blur', 'change'] }],
      proxied: [{ required: true, message: '使用代理', trigger: ['blur', 'change'] }]
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
          {DATA.fromData.record_id ? (
            <ElRow>
              <ElFormItem label="记录ID" prop="record_id">
                <ElInput readonly v-model={DATA.fromData.record_id} placeholder="记录ID" />
              </ElFormItem>
            </ElRow>
          ) : (
            <ElRow>{/*<!--不能不写 {<><></></>}--> */}</ElRow>
          )}

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="记录类型" prop="type">
                <EnumSelect
                  data={api.getAllType()}
                  v-model={DATA.fromData.type}
                  placeholder="记录类型"
                />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="子域名" prop="name">
                <ElInput v-model={DATA.fromData.name} placeholder="子域名: www" />
              </ElFormItem>
            </ElCol>
            <ElCol>
              <ElFormItem label="记录值" prop="content">
                <ElInput v-model={DATA.fromData.content} placeholder="记录值" />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="TTl" prop="ttl">
                <ElInputNumber v-model={DATA.fromData.ttl} placeholder="ttl" />
              </ElFormItem>
            </ElCol>
            {DATA.proxiable ? (
              <ElCol span={12}>
                <ElFormItem label="代理" prop="proxied">
                  <ElSwitch v-model={DATA.fromData.proxied}></ElSwitch>
                </ElFormItem>
              </ElCol>
            ) : (
              <></>
            )}

            <ElCol>
              <ElFormItem label="备注" prop="comment">
                <ElInput v-model={DATA.fromData.comment} type="textarea" placeholder="备注" />
              </ElFormItem>
            </ElCol>
          </ElRow>
        </>
      )
    }

    const rander = (): VNode => {
      return (
        <DialogForm
          closeOnClickModal={false}
          draggable
          title={prop.title}
          labelWidth={prop.labelWidth}
          style={ctx.attrs}
          rules={rules}
          model={DATA.fromData}
          ref={diaglogForm}
          v-slots={fromSlots}
        />
      )
    }

    const openDialog = (oper: FormOperation, item?: api.IListItem) => {
      const data = api.item2param(item)
      DATA.operation = oper
      DATA.proxiable = item?.proxiable ?? true
      DATA.fromData.record_id = data.record_id
      DATA.fromData.name = data.name
      DATA.fromData.type = data.type
      DATA.fromData.content = data.content
      DATA.fromData.ttl = data.ttl

      diaglogForm.value?.openDialog()
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
