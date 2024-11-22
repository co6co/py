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
  type FormData
} from 'co6co'

import { DictSelect, DictSelectInstance } from 'co6co-right'
//import { DictSelect } from 'co6co-right/dist/api/dict/dictType'
import { upload_image_svc, validatorBack, useDictHook } from 'co6co-right'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  type FormRules
} from 'element-plus'

import { DictTypeCodes } from '../../api/app'
import { task as api } from '../../api/biz'

/**
 * npm install vue-codemirror codemirror
 * npm i @codemirror/lang-python
 */
import CodeMirror from 'vue-codemirror6'
import { basicSetup } from 'codemirror'
import { python } from '@codemirror/lang-python'
import { javascript, javascriptLanguage } from '@codemirror/lang-javascript'
import { oneDark } from '@codemirror/theme-one-dark'
import { EditorView, keymap } from '@codemirror/view'
import { tags } from '@lezer/highlight'
import { HighlightStyle } from '@codemirror/language'
import { syntaxHighlighting } from '@codemirror/language'
import { indentWithTab } from '@codemirror/commands'

export interface Item extends FormItemBase {
  id: number
  name: string
  code: string
  /** 任务类型 */
  category: number
  cron: string
  state: number
  /** 源码类型 */
  sourceCategory: number
  sourceCode: string
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
        sourceCategory: 0,
        sourceCode: '',
        execStatus: 0
      },
      testing: false,
      showResult: false,
      testResult: ''
    })

    //@ts-ignore
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', DATA.fromData)
    // 编辑器选项
    let myTheme = EditorView.theme(
      {
        '&': {
          color: 'white',
          backgroundColor: '#034'
        },
        '.cm-content': {
          caretColor: '#0e9'
        },
        '&.cm-focused .cm-cursor': {
          borderLeftColor: '#0e9'
        },
        '&.cm-focused .cm-selectionBackground, ::selection': {
          backgroundColor: '#fff'
        },
        '.cm-gutters': {
          backgroundColor: '#045',
          color: '#ddd',
          border: 'none'
        }
      },
      { dark: true }
    )
    const myHighlightStyle = HighlightStyle.define([
      { tag: tags.keyword, color: '#fc6' },
      { tag: tags.comment, color: '#f5d', fontStyle: 'italic' }
    ])
    const cmOptions = {
      theme: oneDark,
      extensions: [
        basicSetup,
        python(),
        javascript(),
        myTheme,
        syntaxHighlighting(myHighlightStyle),
        keymap.of([indentWithTab])
      ]
    }
    const language = computed(() => {
      if (isPythonCode.value) return python()
      else return javascript()
    })

    const init_data = (oper: FormOperation, item?: Item) => {
      DATA.operation = oper
      DATA.testResult = ''
      DATA.showResult = false
      DATA.testing = false
      switch (oper) {
        case FormOperation.add:
          DATA.id = 0
          DATA.fromData.name = ''

          DATA.fromData.code = ''
          DATA.fromData.category = 0
          DATA.fromData.sourceCategory = 0
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
          DATA.fromData.sourceCategory = item.sourceCategory
          DATA.fromData.cron = item.cron
          DATA.fromData.state = item.state
          DATA.fromData.sourceCode = item.sourceCode
          DATA.fromData.execStatus = item.execStatus
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
    onMounted(() => {})
    onBeforeUnmount(() => {})
    const onRun = () => {
      DATA.testing = true
      DATA.showResult = false
      api
        .test_exe_code_svc(DATA.fromData.sourceCode)
        .then((res) => {
          // CodeMirror 控件对类型敏感如果不是字符串报错卡住整个页面
          if (typeof res.data == 'string') DATA.testResult = res.data
          if (typeof res.data == 'object') DATA.testResult = JSON.stringify(res.data)
          else DATA.testResult = String(res.data)
          DATA.showResult = true
        })
        .finally(() => {
          DATA.testing = false
        })
    }
    const taskCategoryRef = ref<DictSelectInstance>()
    const codeCategoryRef = ref<DictSelectInstance>()

    const allowCorn = computed(() => {
      return taskCategoryRef.value?.flagIs(String(DATA.fromData.category), 'enabled')
    })
    const isPythonCode = computed(() => {
      return codeCategoryRef.value?.flagIs(String(DATA.fromData.sourceCategory), 'python')
    })
    const onImage = upload_image_svc

    const valid = (promise: Promise<IResponse<boolean>>, rule: any, callback: validatorBack) => {
      promise.then((res) => {
        if (res.data) return callback()
        return (rule.message = res.message), callback(new Error(rule.message))
      })
    }
    const validCron = (rule: any, value: any, callback: validatorBack) => {
      valid(api.test_cron2_svc(value), rule, callback)
    }
    const validSourceCode = (rule: any, value: any, callback: validatorBack) => {
      if (isPythonCode.value) valid(api.test_code_svc(value), rule, callback)
      else callback()
    }
    const rules_base: FormRules = {
      name: [{ required: true, message: '请输入名称', trigger: ['blur', 'change'] }],
      code: [{ required: true, message: '请输入编码', trigger: ['blur', 'change'] }],
      category: [{ required: true, message: '请选择类型', trigger: ['blur', 'change'] }],

      state: [{ required: true, message: '状态能为空', trigger: ['blur', 'change'] }],
      sourceCode: [
        { required: true, validator: validSourceCode, message: '源代码', trigger: ['blur'] }
      ]
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
    const rules = computed(() => {
      if (allowCorn.value) {
        return cron_rules
      }
      return rules_base
    })

    //富文本1
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
          {isPythonCode.value ? (
            <ElButton
              disabled={DATA.testing}
              onClick={() => {
                onRun()
              }}
            >
              运行
            </ElButton>
          ) : (
            <></>
          )}
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
                  dictTypeCode={DictTypeCodes.TaskCategory}
                  v-model={DATA.fromData.category}
                  isNumber={true}
                  placeholder="任务类别"
                />
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="源码类型" prop="sourceCategory">
                <DictSelect
                  ref={codeCategoryRef}
                  dictTypeCode={DictTypeCodes.TaskCodeCategory}
                  v-model={DATA.fromData.sourceCategory}
                  isNumber={true}
                  placeholder="源码类型"
                />
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="编码" prop="code">
                <ElInput v-model={DATA.fromData.code} placeholder="编码"></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>
          {allowCorn.value ? (
            <ElRow>
              <ElCol>
                <ElFormItem label="cron表达式" prop="cron">
                  <ElInput v-model={DATA.fromData.cron} placeholder="0 0 0 31 12 ? 2024"></ElInput>
                </ElFormItem>
              </ElCol>
            </ElRow>
          ) : (
            <></>
          )}
          <ElRow>
            <ElCol>
              <ElFormItem label="执行代码" prop="sourceCode">
                <CodeMirror
                  style="width:100%;min-height:100px"
                  v-model={DATA.fromData.sourceCode}
                  dark
                  basic
                  tab
                  tabSize={4}
                  lang={language.value}
                  gutter
                  extensions={cmOptions.extensions}
                />
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="状态" prop="state">
                <DictSelect
                  dictTypeCode={DictTypeCodes.TaskState}
                  v-model={DATA.fromData.state}
                  isNumber={true}
                  placeholder="任务状态"
                />
              </ElFormItem>
            </ElCol>
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
          {DATA.showResult ? (
            <ElRow>
              <ElCol>
                <CodeMirror
                  readonly
                  style="width:100%"
                  v-model={DATA.testResult}
                  dark
                  basic
                  tab
                  tabSize={4}
                  gutter
                  extensions={cmOptions.extensions}
                />
              </ElCol>
            </ElRow>
          ) : (
            <></>
          )}
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
