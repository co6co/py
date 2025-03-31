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
import type { InjectionKey, InputTypeHTMLAttribute } from 'vue'
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

import { DictSelect, DictSelectInstance, validatorBack } from 'co6co-right'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules
} from 'element-plus'

import { DictTypeCodes } from '@/api/app'
import { task_api as api } from '@/api/sys'

export interface Item extends FormItemBase {
  id: number
  name: string
  code: string
  /** 任务类型 */
  category: number
  cron: string
  data: string
  state: number
  execStatus: number
}

//Omit、Pick、Partial、Required
export type FormItem = Omit<Item, 'id' | 'createUser' | 'updateUser' | 'createTime' | 'updateTime'>
export default defineComponent({
  name: 'ModifyTask',
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
    const DATA = reactive<
      FormData<number, FormItem> & { testing: boolean; showResult: boolean; testResult: string }
    >({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        name: '',
        code: '',
        category: 0,
        cron: '',
        state: 0,
        data: '',
        execStatus: 0
      },
      testing: false,
      showResult: false,
      testResult: ''
    })

    //@ts-ignore
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', DATA.fromData)

    const init_data = (oper: FormOperation, item?: Item) => {
      DATA.operation = oper
      DATA.testResult = ''
      DATA.showResult = false
      DATA.testing = false
      switch (oper) {
        case FormOperation.add:
          DATA.id = 0
          const tmp = {
            name: '',
            code: '',
            category: 0,
            cron: '',
            data: '',
            execStatus: 0
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

      onLoadTaskData(DATA.fromData.category)
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
    onMounted(() => {})
    onBeforeUnmount(() => {})

    const taskCategoryRef = ref<DictSelectInstance>()
    const valid = (promise: Promise<IResponse<boolean>>, rule: any, callback: validatorBack) => {
      promise.then((res) => {
        if (res.data) return callback()
        return (rule.message = res.message), callback(new Error(rule.message))
      })
    }
    const validCron = (rule: any, value: any, callback: validatorBack) => {
      valid(api.test_cron2_svc(value), rule, callback)
    }
    const rules_base: FormRules = {
      name: [{ required: true, message: '请输入名称', trigger: ['blur', 'change'] }],
      code: [{ required: true, message: '请输入编码', trigger: ['blur', 'change'] }],
      category: [{ required: true, message: '请选择类型', trigger: ['blur', 'change'] }],
      state: [{ required: true, message: '状态能为空', trigger: ['blur', 'change'] }]
    }
    const cron_rules: FormRules = {
      ...{
        cron: [
          {
            required: true,
            validator: validCron,
            message: 'Cron表达式不正确',
            trigger: ['blur']
          }
        ]
      },
      ...rules_base
    }

    const selectValue = ref<IEnumSelect[]>([])
    const onLoadTaskData = (value: number) => {
      api.get_select_svc(value).then((res) => {
        if (!res.data || res.data.length == 0) selectValue.value = []

        selectValue.value = res.data.map((item: any) => {
          return {
            uid: item.id,
            key: item.id,
            label: item.name,
            value: String(item.id)
          }
        })
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
            <ElCol span={12}>
              <ElFormItem label="名称" prop="name">
                <ElInput v-model={DATA.fromData.name} placeholder="名称"></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="任务类别" prop="category">
                <DictSelect
                  ref={taskCategoryRef}
                  onChange={onLoadTaskData}
                  dictTypeCode={DictTypeCodes.TaskCategory}
                  v-model={DATA.fromData.category}
                  isNumber={true}
                  placeholder="任务类别"
                />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="状态" prop="state">
                <DictSelect
                  dictTypeCode={DictTypeCodes.TaskState}
                  v-model={DATA.fromData.state}
                  isNumber={true}
                  placeholder="状态"
                />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="编码" prop="code">
                <ElInput v-model={DATA.fromData.code} placeholder="编码"></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="cron表达式" prop="cron">
                <ElInput
                  v-model={DATA.fromData.cron}
                  placeholder="59[秒] 59[分] 23[时] 31[日] 12[月] ?[星期] 2024[年]"
                />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="任务对象" prop="data">
                <EnumSelect v-model={DATA.fromData.data} data={selectValue.value} />
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="运行状态" prop="execStatus">
                <DictSelect
                  disabled={true}
                  dictTypeCode={DictTypeCodes.TaskStatus}
                  v-model={DATA.fromData.execStatus}
                  isNumber={true}
                  placeholder="运行状态"
                />
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
          rules={cron_rules}
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
