import { defineComponent, ref, reactive, provide, computed, onMounted } from 'vue'
import type { InjectionKey } from 'vue'
import {
  DialogForm,
  FormOperation,
  showLoading,
  closeLoading,
  type DialogFormInstance,
  type ObjectType,
  type FormData
} from 'co6co'

import { FormItemBase, IResponse } from 'co6co'
import {
  Flag,
  dictSvc as dict_svc,
  useDictHook,
  validatorBack,
  upload_image_svc
} from 'co6co-right'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules,
  ElSelect,
  ElOption,
  ElInputNumber
} from 'element-plus'

import { DictTypeCodes } from '../../api/app'
import { task as api } from '../../api/biz'
import { MdEditor } from 'md-editor-v3'
import 'md-editor-v3/lib/style.css'
export interface Item extends FormItemBase {
  id: number
  name: string
  code: string
  category: number
  cron: string
  state: number
  sourceCode: string
  execStatus: number
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
        cron: '',
        state: 0,
        sourceCode: '',
        execStatus: 0
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
          DATA.fromData.name = ''

          DATA.fromData.code = ''
          DATA.fromData.category = 0
          DATA.fromData.cron = ''
          DATA.fromData.state = 0
          DATA.fromData.sourceCode = ''
          DATA.fromData.execStatus = 0
          break
        case FormOperation.edit:
          if (!item) return false
          DATA.id = item.id

          DATA.fromData.name = item.name
          DATA.fromData.code = item.code
          DATA.fromData.category = item.category
          DATA.fromData.cron = item.cron
          DATA.fromData.state = item.state
          DATA.fromData.sourceCode = item.sourceCode
          DATA.fromData.execStatus = item.execStatus
          //可以在这里写一些use 获取其他的数据
          break
      }
      return true
    }

    const rules: FormRules = {
      name: [{ required: true, message: '请输入名称', trigger: ['blur', 'change'] }],
      code: [{ required: true, message: '请选择类型', trigger: ['blur', 'change'] }],
      category: [{ required: true, message: '请在地图上选择坐标', trigger: ['blur', 'change'] }],
      cron: [{ required: true, message: '请在地图上选择坐标', trigger: ['blur', 'change'] }],
      state: [{ required: true, message: '状态能为空', trigger: ['blur', 'change'] }],
      sourceCode: [{ required: true, message: '排序不能为空', trigger: ['blur', 'change'] }]
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
          ElMessage.success(`操作成功`)
          ctx.emit('saved', res.data)
        })
        .finally(() => {
          closeLoading()
        })
    }
    const useCategory = useDictHook.useDictSelect()
    const useStatue = useDictHook.useDictSelect()
    const useState = useDictHook.useDictSelect()
    const dictState = useDictHook.useState()
    onMounted(async () => {
      await useCategory.queryByCode(DictTypeCodes.TaskCategory)
      await useState.queryByCode(DictTypeCodes.TaskState)
      await useStatue.queryByCode(DictTypeCodes.TaskStatus)
    })
    const onImage = upload_image_svc
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
                <ElSelect v-model={DATA.fromData.category} placeholder="任务类别">
                  {useCategory.selectData.value.map((d, index) => {
                    return <ElOption key={index} label={d.name} value={Number(d.value)}></ElOption>
                  })}
                </ElSelect>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="编码" prop="code">
                <ElInputNumber v-model={DATA.fromData.code} placeholder="编码"></ElInputNumber>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="cron表达式" prop="cron">
                <ElInputNumber
                  v-model={DATA.fromData.cron}
                  placeholder="0 0 0 31 12 ? 2024"
                ></ElInputNumber>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElRow>
            <ElCol span={24}>
              <ElFormItem label="GPS">
                <md-editor class="mgb20" v-model={DATA.fromData.sourceCode} onUploadImg={onImage} />
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="编码" prop="code">
                <ElInputNumber v-model={DATA.fromData.code} placeholder="编码"></ElInputNumber>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="状态" prop="state">
                <ElSelect v-model={DATA.fromData.state} placeholder="请选择">
                  {dictState.selectData.value.map((d, _) => {
                    return <ElOption key={d.key} label={d.label} value={d.value}></ElOption>
                  })}
                </ElSelect>

                <ElSelect v-model={DATA.fromData.category} placeholder="状态1">
                  {useState.selectData.value.map((d, index) => {
                    return <ElOption key={index} label={d.name} value={Number(d.value)}></ElOption>
                  })}
                </ElSelect>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="状态2" prop="state">
                <ElSelect v-model={DATA.fromData.category} placeholder="状态1">
                  {useStatue.selectData.value.map((d, index) => {
                    return <ElOption key={index} label={d.name} value={Number(d.value)}></ElOption>
                  })}
                </ElSelect>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElRow>
            <ElCol>
              <ElFormItem label="说明" prop="description">
                <ElInput
                  v-model={DATA.fromData.sourceCode}
                  type="textarea"
                  autosize={{ minRows: 4, maxRows: 10 }}
                  placeholder="说明"
                ></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>
        </>
      )
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
