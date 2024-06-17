import { ref, reactive, defineComponent, defineExpose } from 'vue'

import {
  ElDialog,
  ElForm,
  ElFormItem,
  ElSelect,
  ElOption,
  ElSwitch,
  ElInputNumber,
  ElMessage,
  ElButton,
  type FormInstance,
  type FormRules,
  ElInput,
  ElCol,
  ElRow
} from 'element-plus'
import * as api from '../../api/boat/rules'
import * as types from 'co6co'
import useNotifyAudit, { NotifyType } from '../../hook/useNotifyAudit'
import { CircleCheck, CircleClose } from '@element-plus/icons-vue'
export interface Item {
  id: number
  name: string //: "\u6d4b\u8bd5\u89c4\u5219",
  enable: number //: 0,
  code: string //: null,
  confidence: number //: 0,
  priority: number //: 99999,
  baseRule: number //: 0,
  objNumLimit: number //: 1,
  objNumLowerLimit: number //: 0,
  objNumUpperLimit: number //: 0,
  bnessRule: number //: 0,
  bnessLimit: number //: 1,
  aiNetName: string //: null,
  /**
   * 人工复审
   */
  manualReview: number //: 0, //将 isAiAudit 移出
  remark: string //: null,
  createTime: string //: null,
  updateTime: string //: null,
  createUser: number //: null,
  updateUser: number //: null
  createUserName: string //: null
  aiAuditAccuracy: number
}

//Omit、Pick、Partial、Required
export type FromData = Omit<
  Item,
  | 'id'
  | 'createUserName'
  | 'createTime'
  | 'updateTime'
  | 'createUser'
  | 'updateUser'
  | 'createUserName'
>
export interface dialogDataType {
  visible: boolean
  operation: types.Operation
  title?: string
  id: number
  loading: boolean
  fromData: FromData
}

export default defineComponent({
  name: 'EditUserBoat',
  emits: {
    saved: (saveData: FromData) => true
  },

  setup(_, context) {
    const dialogForm = ref<FormInstance>()
    const rules: FormRules = {
      name: [{ required: true, message: '规则名称', trigger: ['blur'] }],
      enable: [{ required: true, message: '是否启用', trigger: ['blur'] }],
      code: [{ required: true, message: '代码', trigger: ['blur'] }],
      confidence: [{ required: true, message: '可行度', trigger: ['blur'] }],
      priority: [{ required: true, message: '优先级', trigger: ['blur'] }],
      baseRule: [{ required: true, message: '基础规则', trigger: ['blur'] }],
      objNumLimit: [{ required: true, message: '目标数量', trigger: ['blur'] }],
      objNumLowerLimit: [{ required: true, message: '目标数量下限', trigger: ['blur'] }],
      objNumUpperLimit: [{ required: true, message: '目标数量上限', trigger: ['blur'] }],
      bnessRule: [{ required: true, message: '业务规则', trigger: ['blur'] }],
      bnessLimit: [{ required: true, message: '业务目标数量', trigger: ['blur'] }],
      aiNetName: [{ required: true, message: '名称', trigger: ['blur'] }],
      isAiAudit: [{ required: true, message: '是AI审核', trigger: ['blur'] }]
      //remark: [{ required: true, message: '输入备注', trigger: ['blur'] }]
    }

    const form = reactive<dialogDataType>({
      visible: false,
      operation: types.Operation.Add,
      id: 0,
      loading: false,
      fromData: {
        name: '',
        enable: 0,
        code: '',
        confidence: 0,
        priority: 0,
        baseRule: 0,
        objNumLimit: 0,
        objNumLowerLimit: 0,
        objNumUpperLimit: 0,
        bnessRule: 0,
        bnessLimit: 0,
        aiNetName: '',
        manualReview: 0,
        aiAuditAccuracy: 0,
        remark: ''
      }
    })
    const { setOldState, notifyAuditSystem } = useNotifyAudit()
    const onNotifyAuditSystem = () => {
      notifyAuditSystem({
        type: NotifyType.manual_review_state_changed,
        state: form.fromData.manualReview
      })
    }

    //其他api 操作
    //end
    const onOpenDialog = (operation: types.Operation.Add | types.Operation.Edit, item?: Item) => {
      form.visible = true
      form.operation = operation
      form.id = -1
      setOldState(item?.manualReview)
      switch (operation) {
        case types.Operation.Add:
          form.title = '增加'

          form.fromData.name = ''
          form.fromData.enable = 0
          form.fromData.code = ''
          form.fromData.confidence = 0
          form.fromData.priority = 0
          form.fromData.baseRule = 0
          form.fromData.objNumLimit = 0
          form.fromData.objNumLowerLimit = 0
          form.fromData.objNumUpperLimit = 0
          form.fromData.bnessRule = 0
          form.fromData.bnessLimit = 0
          form.fromData.aiNetName = ''
          form.fromData.manualReview = 0
          form.fromData.aiAuditAccuracy = 0
          form.fromData.remark = ''

          break
        case types.Operation.Edit:
          if (item && item.id) {
            form.id = item.id
            form.title = '编辑'
            form.fromData.name = item.name
            form.fromData.enable = item.enable
            form.fromData.code = item.code
            form.fromData.confidence = item.confidence
            form.fromData.priority = item.priority
            form.fromData.baseRule = item.baseRule
            form.fromData.objNumLimit = item.objNumLimit
            form.fromData.objNumLowerLimit = item.objNumLowerLimit
            form.fromData.objNumUpperLimit = item.objNumUpperLimit
            form.fromData.bnessRule = item.bnessRule
            form.fromData.bnessLimit = item.bnessLimit
            form.fromData.aiNetName = item.aiNetName
            form.fromData.manualReview = item.manualReview
            form.fromData.aiAuditAccuracy = item.aiAuditAccuracy
            form.fromData.remark = item.remark
          }
          break
      }
    }
    const save = (formEl: FormInstance | undefined) => {
      if (!formEl) return
      formEl.validate((value) => {
        if (value) {
          form.loading = true
          if (form.operation == types.Operation.Add) {
            api
              .add_svc(form.fromData)
              .then((res) => {
                if (res.code == 0) {
                  form.visible = false
                  ElMessage.success(`增加成功`)
                  context.emit('saved', form.fromData)
                  onNotifyAuditSystem()
                } else {
                  ElMessage.error(`增加失败:${res.message}`)
                }
              })
              .finally(() => {
                form.loading = false
              })
          } else {
            api
              .edit_svc(form.id, form.fromData)
              .then((res) => {
                if (res.code == 0) {
                  form.visible = false
                  ElMessage.success(`编辑成功`)
                  context.emit('saved', form.fromData)
                  onNotifyAuditSystem()
                } else {
                  ElMessage.error(`编辑失败:${res.message}`)
                }
              })
              .finally(() => {
                form.loading = false
              })
          }
        } else {
          ElMessage.error('请检查输入的数据！')
          return false
        }
      })
    }

    context.expose({
      onOpenDialog
    })
    return () => {
      //可以写某些代码
      return (
        <>
          <ElDialog
            title={form.title}
            v-model={form.visible}
            v-slots={{
              footer: () => {
                return (
                  <span class="dialog-footer">
                    <ElButton
                      onClick={() => {
                        console.info(form.visible)
                        form.visible = false
                      }}
                    >
                      关闭
                    </ElButton>
                    <ElButton
                      loading={form.loading}
                      disabled={form.loading}
                      onClick={() => {
                        save(dialogForm.value)
                      }}
                    >
                      保存
                    </ElButton>
                  </span>
                )
              }
            }}
          >
            <ElForm labelWidth={120} ref={dialogForm} rules={rules} model={form.fromData}>
              <ElFormItem label="规则名称" prop="userId">
                <ElInput
                  style="width: 160px"
                  clearable={true}
                  v-model={form.fromData.name}
                  placeholder="规则名称"
                ></ElInput>
              </ElFormItem>
              <ElFormItem label="是否启用" prop="enable">
                <ElSwitch
                  v-model={form.fromData.enable}
                  activeValue={1}
                  inactiveValue={0}
                  inactiveColor="#b6c0c7"
                  activeActionIcon={CircleCheck}
                  inactiveActionIcon={CircleClose}
                ></ElSwitch>
              </ElFormItem>
              <ElRow>
                <ElCol span={12}>
                  <ElFormItem label="规则代码" prop="code">
                    <ElInput
                      style="width: 160px"
                      clearable={true}
                      v-model={form.fromData.code}
                      placeholder="代码"
                    ></ElInput>
                  </ElFormItem>
                </ElCol>
                <ElCol span={12}>
                  <ElFormItem label="AI审核正确率" prop="aiAuditAccuracy">
                    <ElInputNumber
                      v-model={form.fromData.aiAuditAccuracy}
                      placeholder="AI审核正确率"
                    ></ElInputNumber>
                  </ElFormItem>
                </ElCol>
              </ElRow>

              <ElRow>
                <ElCol span={12}>
                  <ElFormItem label="AI网络模型名称" prop="aiNetName">
                    <ElInput
                      style="width: 160px"
                      clearable={true}
                      v-model={form.fromData.aiNetName}
                      placeholder="AI网络模型名称"
                    ></ElInput>
                  </ElFormItem>
                </ElCol>
                <ElCol span={12}>
                  <ElFormItem label="人工复审" prop="manualRevlew">
                    <ElSwitch
                      v-model={form.fromData.manualReview}
                      activeValue={1}
                      inactiveValue={0}
                      inactiveColor="#b6c0c7"
                      activeActionIcon={CircleCheck}
                      inactiveActionIcon={CircleClose}
                    ></ElSwitch>
                  </ElFormItem>
                </ElCol>
              </ElRow>

              <ElRow>
                <ElCol span={12}>
                  <ElFormItem label="可信度" prop="confidence">
                    <ElInputNumber
                      min={0}
                      max={9999}
                      v-model={form.fromData.confidence}
                      placeholder="可行度"
                    ></ElInputNumber>
                  </ElFormItem>
                </ElCol>

                <ElCol span={12}>
                  <ElFormItem label="优先级" prop="priority">
                    <ElInputNumber
                      min={0}
                      max={99999}
                      v-model={form.fromData.priority}
                      placeholder="优先级"
                    ></ElInputNumber>
                  </ElFormItem>
                </ElCol>
              </ElRow>

              <ElFormItem label="基础规则" prop="baseRule">
                <ElInput v-model={form.fromData.baseRule} placeholder="基础规则"></ElInput>
              </ElFormItem>

              <ElFormItem label="目标数量" prop="objNumLimit">
                <ElInputNumber
                  v-model={form.fromData.objNumLimit}
                  placeholder="目标数量"
                ></ElInputNumber>
              </ElFormItem>

              <ElRow>
                <ElCol span={12}>
                  <ElFormItem label="目标上下限" prop="objNumLowerLimit">
                    <ElInputNumber
                      v-model={form.fromData.objNumLowerLimit}
                      placeholder="目标上下限"
                    ></ElInputNumber>
                  </ElFormItem>
                </ElCol>

                <ElCol span={12}>
                  <ElFormItem label="目标上限" prop="objNumUpperLimit">
                    <ElInputNumber
                      v-model={form.fromData.objNumUpperLimit}
                      placeholder="目标上限"
                    ></ElInputNumber>
                  </ElFormItem>
                </ElCol>
              </ElRow>

              <ElRow>
                <ElCol span={12}>
                  <ElFormItem label="业务规则" prop="bnessRule">
                    <ElInputNumber
                      v-model={form.fromData.bnessRule}
                      placeholder="业务规则"
                    ></ElInputNumber>
                  </ElFormItem>
                </ElCol>
                <ElCol span={12}>
                  <ElFormItem label="业务目标数量" prop="bnessLimit">
                    <ElInputNumber
                      v-model={form.fromData.bnessLimit}
                      placeholder="业务目标数量"
                    ></ElInputNumber>
                  </ElFormItem>
                </ElCol>
              </ElRow>

              <ElFormItem label="备注" prop="remark">
                <ElInput
                  type="textarea"
                  clearable={true}
                  v-model={form.fromData.remark}
                  placeholder="备注"
                ></ElInput>
              </ElFormItem>
            </ElForm>
          </ElDialog>
        </>
      )
    }
  }
})
