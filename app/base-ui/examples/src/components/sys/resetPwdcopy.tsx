import { defineComponent, ref, reactive, computed, type PropType, provide, inject } from 'vue'
import type { InjectionKey } from 'vue'
import { EcDialogForm } from 'co6co'
import { type ObjectType, type FormData, FormOperation } from 'co6co'
import * as api_type from 'co6co'
import { retsetPwd_svc } from '../../api/sys/user'
import { useTree } from '../../hook/sys/useUserGroupSelect'
import { useState } from '../../hook/sys/useUserSelect'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInputNumber,
  ElInput,
  ElMessage,
  type FormRules,
  type FormInstance,
  ElSelect,
  ElOption,
  ElTreeV2,
  ElTreeSelect,
  ElTree
} from 'element-plus'
import { Plus, Minus, Operation } from '@element-plus/icons-vue'
import { showLoading, closeLoading } from '../Logining'

export interface Item extends api_type.FormItemBase {
  userName?: string
  password?: string
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
      default: 110
    }
  },
  emits: {
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const { treeSelectData, refresh } = useTree(0)
    const { selectData } = useState()
    const diaglogForm = ref<InstanceType<typeof EcDialogForm>>()
    const DATA = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: { userName: '' }
    })
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', DATA.fromData)

    const init_data = (item?: Item) => {
      DATA.operation = api_type.FormOperation.edit
      if (!item) return false
      DATA.fromData.userName = item.userName
      DATA.fromData.password = ''
      return true
    }
    const rules_base: FormRules = {
      userName: [{ required: true, message: '请输入用户名', trigger: ['blur'] }],
      password: [
        {
          required: true,
          min: 6,
          max: 20,
          message: '请输入6-20位新密码',
          trigger: ['blur', 'change']
        }
      ]
    }

    const rules = computed(() => {
      return rules_base
    })
    const save = () => {
      //提交数据
      let promist: Promise<api_type.IResponse> = retsetPwd_svc(DATA.fromData)
      showLoading()
      promist
        .then((res) => {
          if (res.code == 0) {
            diaglogForm.value?.closeDialog()
            ElMessage.success(`重置密码成功`)
            refresh()
            ctx.emit('saved', res.data)
          } else {
            ElMessage.error(`重置密码失败:${res.message}`)
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
          <ElFormItem label="用户名" prop="userName">
            <ElInput v-model={DATA.fromData.userName} readonly></ElInput>
          </ElFormItem>
          <ElFormItem label="新密码" prop="password">
            <ElInput type="password" v-model={DATA.fromData.password} show-password></ElInput>
          </ElFormItem>
        </>
      )
    }

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
    const openDialog = (item?: Item) => {
      init_data(item)
      diaglogForm.value?.openDialog()
    }
    const update = () => {
      refresh()
    }
    ctx.expose({
      openDialog,
      update
    })
    rander.openDialog = openDialog
    rander.update = update
    return rander
  } //end setup
})
