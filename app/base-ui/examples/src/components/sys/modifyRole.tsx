import { defineComponent, ref, reactive, type PropType, provide, inject, computed } from 'vue'
import type { InjectionKey } from 'vue'
import { EcDialogForm } from 'co6co'
import { type ObjectType, type FormData, FormOperation } from 'co6co'
import * as api_type from 'co6co'
import api from '../../api/sys/role'

import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules,
  type FormInstance,
  ElSelect,
  ElTreeV2,
  ElTreeSelect,
  ElTree,
  ElOption,
  ElSelectV2,
  ElInputNumber
} from 'element-plus'
import { Plus, Minus } from '@element-plus/icons-vue'
import { showLoading, closeLoading } from '../Logining'

export interface Item extends api_type.FormItemBase {
  id: number
  name?: string
  code?: string
  order?: number
  remark?: string
}
//Omit、Pick、Partial、Required
export type FormItem = Omit<Item, 'id' | 'createUser' | 'updateUser' | 'createTime' | 'updateTime'>
export default defineComponent({
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
    const diaglogForm = ref<InstanceType<typeof EcDialogForm>>()
    const DATA = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: {}
    })
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', DATA.fromData)

    const init_data = (oper: FormOperation, item?: Item) => {
      DATA.operation = oper
      switch (oper) {
        case FormOperation.add:
          DATA.id = -1
          DATA.fromData.name = ''
          DATA.fromData.code = ''
          DATA.fromData.order = 1
          DATA.fromData.remark = ''
          break
        case FormOperation.edit:
          if (!item) return false
          DATA.id = item.id
          DATA.fromData.name = item.name
          DATA.fromData.code = item.code

          DATA.fromData.order = item.order
          DATA.fromData.remark = item.remark
          //可以在这里写一些use 获取其他的数据
          break
      }
      return true
    }
    const rules_b: FormRules = {
      name: [{ required: true, message: '请输入角色名', trigger: 'blur' }],
      code: [{ required: true, message: '请输入角色编码', trigger: 'blur' }],
      order: [{ required: true, message: '请输入排序编号', trigger: 'blur,change' }]
    }
    const save = () => {
      //提交数据
      let promist: Promise<api_type.IResponse>
      switch (DATA.operation) {
        case FormOperation.add:
          promist = api.add_svc(DATA.fromData)
          break
        case FormOperation.edit:
          //提交前是否应该删除其他类别的数据
          promist = api.edit_svc(DATA.id, DATA.fromData)
          break
        default:
          //没有相关操作
          return
      }
      showLoading()
      promist
        .then((res) => {
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
            <ElCol span={12}></ElCol>
            <ElCol span={12}></ElCol>
          </ElRow>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="名称" prop="name">
                <ElInput v-model={DATA.fromData.name} placeholder="用户组名称"></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="代码" prop="code">
                <ElInput v-model={DATA.fromData.code} placeholder="用户组代码"></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol>
              <ElFormItem label="排序" prop="order">
                <ElInputNumber v-model={DATA.fromData.order} placeholder="排序"></ElInputNumber>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElFormItem label="备注" prop="remark">
            <ElInput
              type="textarea"
              clearable={true}
              v-model={DATA.fromData.remark}
              placeholder="备注"
            ></ElInput>
          </ElFormItem>
        </>
      )
    }

    const rules = computed(() => {
      return rules_b
    })
    const rander = (): ObjectType => {
      return (
        <EcDialogForm
          title={prop.title}
          labelWidth={prop.labelWidth}
          style={ctx.attrs}
          rules={rules.value}
          ref={diaglogForm}
          v-slots={fromSlots}
        ></EcDialogForm>
      )
    }
    const openDialog = (oper: FormOperation, item?: Item) => {
      init_data(oper, item)
      diaglogForm.value?.openDialog()
    }
    const update = () => {}
    ctx.expose({
      openDialog,
      update
    })
    rander.openDialog = openDialog
    rander.update = update
    return rander
  } //end setup
})
