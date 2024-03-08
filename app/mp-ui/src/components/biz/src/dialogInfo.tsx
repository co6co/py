import { ref, reactive, defineComponent, defineExpose } from 'vue'

import { 
  ElDialog,
  ElForm,
  ElFormItem,
  ElCard,
  ElSelect,
  ElOption,
  ElSwitch,
  ElInputNumber,
  ElMessage,
  ElButton,
  ElRadioGroup,
  ElRadio,
  type FormInstance,
  type FormRules,
  ElInput,
  ElCol,
  ElRow
} from 'element-plus' 
import { showLoading, closeLoading } from '../../components/Logining' 

enum PriorityType {
  boat,
  time,
  rule
}
export interface dialogDataType {
  title?: string
  loading: boolean
  fromData: api.IRecode & { priorityOption: PriorityType }
}

export default defineComponent({
  name: 'settingAudit',
  setup(_, context) {
    const dialogForm = ref<FormInstance>()
    const rules: FormRules = {
      expiration_time: [
        {
          required: true,
          type: 'number',
          message: '过期时间为数字，单位小时',
          trigger: ['blur', 'change']
        }
      ]
    }

    const form = reactive<dialogDataType>({
      loading: false,
      fromData: {
        ai_audited_no_review: 0,
        expiration_time: 0,
        priorityOption: PriorityType.boat,
        priority: {
          boat_priority: 0,
          latest_time_priority: 0,
          rule_priority: 0
        }
      }
    })

    //其他api 操作
    const getData = async () => {
      showLoading()
      api
        .get_svc()
        .then((res) => {
          if (pd_api.isSuccess(res)) { 
            form.fromData = { ...res.data, ...{ priorityOption:  to_PriorityType(res.data) } }
          }
        }).catch((e)=>{
          ElMessage.error(`加载数据失败，请刷新重试！${e.message}`)
        })
        .finally(() => {
          closeLoading()
        })
    }
    getData()
    const to_PriorityType = (data: api.IRecode) => {
      if (data.priority.boat_priority) return PriorityType.boat
      else if (data.priority.latest_time_priority) return PriorityType.time
      else if (data.priority.rule_priority) return PriorityType.rule
      return PriorityType.boat
    }
    const to_parm = () => {
      const { priorityOption, ...param } = form.fromData ;
      switch (priorityOption) {
        case PriorityType.boat:
          param.priority.boat_priority = 1
          param.priority.latest_time_priority = 0
          param.priority.rule_priority = 0
          break
        case PriorityType.time:
          param.priority.boat_priority = 0
          param.priority.latest_time_priority = 1
          param.priority.rule_priority = 0
          break
        case PriorityType.rule:
          param.priority.boat_priority = 0
          param.priority.latest_time_priority = 0
          param.priority.rule_priority = 1
          break
      }
      return param
    }
    //end
    const save = (formEl: FormInstance | undefined) => {
      if (!formEl) return
      formEl.validate((value) => {
        if (value) {
          form.loading = true 
          api
            .set_svc({ data: to_parm()  })
            .then((res) => {
              if (pd_api.isSuccess(res)) {
                ElMessage.success(`编辑成功`)
                getData()
              } else {
                ElMessage.error(`编辑失败:${res.message}`)
              }
            })
            .finally(() => {
              form.loading = false
            })
        } else {
          ElMessage.error('请检查输入的数据！')
          return false
        }
      })
    }
    return () => {
      //可以写某些代码
      return (
        <>
          <ElCard>
            <ElForm labelWidth={150} ref={dialogForm} rules={rules} model={form.fromData}>
              <ElFormItem label="Ai审核需人工审核" prop="ai_audited_no_review">
                <ElSwitch
                  v-model={form.fromData.ai_audited_no_review}
                  activeValue={1}
                  inactiveValue={0}
                ></ElSwitch>
              </ElFormItem>

              <ElFormItem label="过期时间" prop="expiration_time">
                <ElCol span={5}>
                  <ElInput
                    v-model={form.fromData.expiration_time}
                    parser={(v: string) => {
                      return Number(v)
                    }}
                    onChange={(v: string) =>
                      (form.fromData.expiration_time = Number(v.replace(/[^\d]/g, '')))
                    }
                    v-slots={{
                      suffix: () => {
                        return '小时'
                      }
                    }}
                  ></ElInput>
                </ElCol>
              </ElFormItem>

              <ElFormItem label="优先级" prop="priorityOption">
                <ElRadioGroup v-model={form.fromData.priorityOption}>
                  <ElRadio label={PriorityType.boat}>船舶优先</ElRadio>
                  <ElRadio label={PriorityType.time}>最新时间优先</ElRadio>
                  <ElRadio label={PriorityType.rule}>业务规则优先</ElRadio>
                </ElRadioGroup>
              </ElFormItem>

              <ElFormItem>
                <ElButton type="primary" onClick={() => save(dialogForm.value)}>
                  保存
                </ElButton>
                <ElButton
                  onClick={() => {
                    getData()
                  }}
                >
                  刷新
                </ElButton>
              </ElFormItem>
            </ElForm>
          </ElCard>
        </>
      )
    }
  }
})
