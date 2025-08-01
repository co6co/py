import {
  defineComponent,
  ref,
  reactive,
  computed,
  provide,
  onMounted,
  onBeforeUnmount,
  VNode
} from 'vue'
import type { InjectionKey } from 'vue'
import {
  DialogForm,
  FormOperation,
  showLoading,
  closeLoading,
  FormItemBase,
  IResponse,
  type DialogFormInstance,
  type FormData,
  IEnumSelect,
  EnumSelect
} from 'co6co'

import { DictSelect, DictSelectInstance, validatorBack, StateSelect } from 'co6co-right'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  ElSelect,
  ElOption,
  type FormRules
} from 'element-plus'

import * as api from '@/api/dev'


export interface Item extends FormItemBase {
  id: number
  name: string
  serialNumber?:string
  category: number
  code: string
  ip: string
  lat: string
  lng: string
  userName: string
  passwd: string
  state: number,
  checkState?:string,
  checkDesc?:string,
  checkTime?:string
  checkImgPath?:string
  vender?:string 
}

//Omit、Pick、Partial、Required
export type FormItem = Omit<Item, 'id' | 'createUser' | 'updateUser' | 'createTime' | 'updateTime'>
export default defineComponent({
  name: 'ModifyDEV',
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
    const DATA = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        name: '',
        code: '',
        category: 0,

        state: 0,
        ip: '',
        lat: '',
        lng: '',
        userName: '',
        passwd: ''
      }
    })

    //@ts-ignore
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', DATA.fromData)

    const init_data = (oper: FormOperation, item?: Item) => {
      DATA.operation = oper

      switch (oper) {
        case FormOperation.add:
          DATA.id = 0
          const tmp: FormItem = {
            name: '',
            code: '',
            category: 0,
            state: 0,
            ip: '',
            lat: '',
            lng: '',
            userName: '',
            passwd: ''
          }
          Object.assign(DATA.fromData, tmp)

          break
        case FormOperation.edit:
          if (!item) return false
          DATA.id = item.id
          Object.assign(DATA.fromData, item)
          //可以在这里写一些use 获取其他的数据
          break
      }
      return true
    }

    const save = () => {
      //提交数据
      let promist: Promise<IResponse>
      switch (DATA.operation) {
        case FormOperation.add:
          promist = api.add_svc(DATA.fromData)
          break
        case FormOperation.edit:
          promist = api.edit_svc(DATA.id, DATA.fromData)
          break
        default:
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
    const DeviceCategory = ref<IEnumSelect[]>([])

    onMounted(async () => {
      const res = await api.dev_category_svc()
      DeviceCategory.value = res.data
    })

    onBeforeUnmount(() => {})

    const rules: FormRules = {
      name: [{ required: true, message: '请输入名称', trigger: ['blur', 'change'] }],
      code: [{ required: true, message: '请输入编码', trigger: ['blur', 'change'] }],
      category: [{ required: true, message: '设备类型', trigger: ['blur', 'change'] }],
      ip: [{ required: true, message: '请输入网络地址', trigger: ['blur', 'change'] }],
      state: [{ required: true, message: '状态能为空', trigger: ['blur', 'change'] }]
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
          <ElRow>
            <ElCol span={24}>
              <ElFormItem label="名称" prop="name">
                <ElInput v-model={DATA.fromData.name} placeholder="名称"></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="编码" prop="code">
                <ElInput v-model={DATA.fromData.code} placeholder="编码"></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="设备类型" prop="category">
                <EnumSelect
                  data={DeviceCategory.value}
                  v-model={DATA.fromData.category}
                  placeholder="设备类型"
                />
              </ElFormItem>
            </ElCol>

            <ElCol span={12}>
              <ElFormItem label="网络地址" prop="ip">
                <ElInput v-model={DATA.fromData.ip} />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="用户名" prop="userName">
                <ElInput v-model={DATA.fromData.userName} />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="密码" prop="passwd">
                <ElInput v-model={DATA.fromData.passwd} />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="经度" prop="lng">
                <ElInput v-model={DATA.fromData.lng} />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="维度" prop="lat">
                <ElInput v-model={DATA.fromData.lat} />
              </ElFormItem>
            </ElCol>

            <ElCol span={12}>
              <ElFormItem label="状态" prop="state">
                <StateSelect v-model={DATA.fromData.state} placeholder="状态" />
              </ElFormItem>
            </ElCol>
          </ElRow>
        </>
      )
    }

    const rander = (): VNode => {
      return (
        <DialogForm
          title={prop.title}
          labelWidth={prop.labelWidth}
          style={ctx.attrs}
          rules={rules}
          closeOnClickModal={false}
          draggable
          ref={diaglogForm}
          v-slots={fromSlots}
        />
      )
    }
    const openDialog = (oper: FormOperation, item?: Item) => {
      init_data(oper, item)
      diaglogForm.value?.openDialog()
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
