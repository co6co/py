import { ref, reactive, defineComponent, computed, defineExpose } from 'vue'

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
import * as pd_api from '../../api/pd'
import * as api from '../../api/pd/auditconfig'
import { showLoading, closeLoading, type ISelect } from 'co6co'

import useUserSelect from '../../hook/useUserSelect'
import { usePermission, ViewFeature, getCurrentRoute } from '../../hook/sys/useRoute'

enum PriorityType {
  ai = 1,
  boat,
  time,
  rule
}
export interface dialogDataType {
  title?: string
  loading: boolean
  priorityUIEles: {
    level: api.PriorityLevel
    name: string
    prop: string
    priorityOption: api.PriorityValue
  }[]
  fromData: api.IRecode
}

export default defineComponent({
  name: 'settingAudit',
  setup(_, context) {
    const dialogForm = ref<FormInstance>()
    const { getPermissKey } = usePermission()
    const rules: FormRules = {
      user_id: [
        {
          required: true,
          type: 'number',
          message: '请选择设置用户',
          trigger: ['blur', 'change']
        }
      ],
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
      priorityUIEles: [
        { level: 1, name: '一级优先级', prop: 'priorityOption1', priorityOption: PriorityType.ai },
        {
          level: 2,
          name: '二级优先级',
          prop: 'priorityOption2',
          priorityOption: PriorityType.boat
        },
        {
          level: 3,
          name: '三级优先级',
          prop: 'priorityOption3',
          priorityOption: PriorityType.time
        },
        { level: 4, name: '四级优先级', prop: 'priorityOption4', priorityOption: PriorityType.rule }
      ],
      fromData: {
        user_id: -1,
        ai_audited_no_review: 0,
        expiration_time: 0,
        priority: {
          ai_pass_priority: 0,
          boat_priority: 0,
          latest_time_priority: 0,
          rule_priority: 0
        }
      }
    })

    //其他api 操作
    const getData = async () => {
      if (form.fromData.user_id <= 0) {
        ElMessage.warning('请选择用户')
        return
      }
      showLoading()
      api
        .get_svc(form.fromData.user_id)
        .then((res) => {
          if (pd_api.isSuccess(res)) {
            to_PriorityType(res.data)
            let userId = form.fromData.user_id
            form.fromData = { ...res.data }
            if (res.data.user_id == undefined)
              console.warn('接口数据没有user_id属性！'), (form.fromData.user_id = userId)
          }
        })
        .catch((e) => {
          ElMessage.error(`加载数据失败，请刷新重试！${e.message}`)
        })
        .finally(() => {
          closeLoading()
        })
    }
    const _setp = (v: number, result: PriorityType) => {
      if (v > 0 && v <= 4) {
        form.priorityUIEles[v - 1].priorityOption = result
      } else if (v > 4) console.warn('未知优先级')
    }
    const to_PriorityType = (data: api.IRecode) => {
      _setp(data.priority.ai_pass_priority, PriorityType.ai)
      _setp(data.priority.boat_priority, PriorityType.boat)
      _setp(data.priority.latest_time_priority, PriorityType.time)
      _setp(data.priority.rule_priority, PriorityType.rule)
    }
    const onLevel = (level: api.PriorityLevel, type: PriorityType) => {
      form.priorityUIEles[level - 1].priorityOption = type
      for (let i = level; i < 4; i++) {
        if (form.priorityUIEles[i].priorityOption == type) form.priorityUIEles[i].priorityOption = 0
      }
      switch (type) {
        case PriorityType.ai:
          form.fromData.priority.ai_pass_priority = level
          if (form.fromData.priority.boat_priority == level)
            form.fromData.priority.boat_priority = 0
          if (form.fromData.priority.latest_time_priority == level)
            form.fromData.priority.latest_time_priority = 0
          if (form.fromData.priority.rule_priority == level)
            form.fromData.priority.rule_priority = 0
          break
        case PriorityType.boat:
          form.fromData.priority.boat_priority = level
          if (form.fromData.priority.ai_pass_priority == level)
            form.fromData.priority.ai_pass_priority = 0
          if (form.fromData.priority.latest_time_priority == level)
            form.fromData.priority.latest_time_priority = 0
          if (form.fromData.priority.rule_priority == level)
            form.fromData.priority.rule_priority = 0
          break
        case PriorityType.time:
          form.fromData.priority.latest_time_priority = level
          if (form.fromData.priority.ai_pass_priority == level)
            form.fromData.priority.ai_pass_priority = 0
          if (form.fromData.priority.boat_priority == level)
            form.fromData.priority.boat_priority = 0
          if (form.fromData.priority.rule_priority == level)
            form.fromData.priority.rule_priority = 0
          break
        case PriorityType.rule:
          form.fromData.priority.rule_priority = level
          if (form.fromData.priority.ai_pass_priority == level)
            form.fromData.priority.ai_pass_priority = 0
          if (form.fromData.priority.boat_priority == level)
            form.fromData.priority.boat_priority = 0
          if (form.fromData.priority.latest_time_priority == level)
            form.fromData.priority.latest_time_priority = 0
          break
      }
    }

    const elDisabled = computed(() => {
      return function (level: api.PriorityLevel, value: api.PriorityValue) {
        switch (level) {
          case 1:
            return false
          case 2:
            return value == 1
          case 3:
            return value == 2 || value == 1
          case 4:
            return value == 3 || value == 2 || value == 1
        }
      }
    })
    const { userSelect } = useUserSelect()
    //end
    const save = (formEl: FormInstance | undefined) => {
      if (!formEl) return
      formEl.validate((isValid) => {
        if (isValid) {
          form.loading = true
          api
            .set_svc({ data: form.fromData })
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
          return Promise.reject('valid Form Error')
        }
      })
    }
    return () => {
      //可以写某些代码
      return (
        <>
          <ElCard>
            <ElForm labelWidth={150} ref={dialogForm} rules={rules} model={form.fromData}>
              <ElFormItem label="关联用户" prop="user_id">
                <ElSelect
                  style="width: 160px"
                  clearable={true}
                  v-model={form.fromData.user_id}
                  onChange={getData}
                  placeholder="请选择"
                >
                  <ElOption label="--请选择--" value={-1}></ElOption>
                  {userSelect.value.map((item: ISelect, index: number) => {
                    return <ElOption key={index} label={item.name} value={item.id}></ElOption>
                  })}
                </ElSelect>
              </ElFormItem>

              <ElFormItem label="Ai审核需人工审核" prop="ai_audited_no_review">
                <ElSwitch
                  v-model={form.fromData.ai_audited_no_review}
                  activeValue={0}
                  inactiveValue={1}
                ></ElSwitch>
              </ElFormItem>

              <ElFormItem label="过期时间" prop="expiration_time">
                <ElCol span={5}>
                  <ElInput
                    v-model={form.fromData.expiration_time}
                    formatter={(value: any) => {
                      return value
                    }}
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
              {form.priorityUIEles.map((item, index) => {
                return (
                  <ElFormItem label={item.name} prop={item.prop}>
                    <ElRadioGroup
                      v-model={item.priorityOption}
                      onChange={(v: any) => {
                        onLevel(item.level, v)
                      }}
                    >
                      <ElRadio
                        disabled={elDisabled.value(
                          item.level,
                          form.fromData.priority.ai_pass_priority
                        )}
                        label={PriorityType.ai}
                      >
                        AI审核通过优先
                      </ElRadio>
                      <ElRadio
                        disabled={elDisabled.value(
                          item.level,
                          form.fromData.priority.boat_priority
                        )}
                        label={PriorityType.boat}
                      >
                        船舶优先
                      </ElRadio>
                      <ElRadio
                        disabled={elDisabled.value(
                          item.level,
                          form.fromData.priority.latest_time_priority
                        )}
                        label={PriorityType.time}
                      >
                        最新时间优先
                      </ElRadio>
                      <ElRadio
                        disabled={elDisabled.value(
                          item.level,
                          form.fromData.priority.rule_priority
                        )}
                        label={PriorityType.rule}
                      >
                        业务规则优先
                      </ElRadio>
                    </ElRadioGroup>
                  </ElFormItem>
                )
              })}

              <ElFormItem>
                <ElButton
                  type="primary"
                  v-permiss={getPermissKey(ViewFeature.setting)}
                  onClick={() => save(dialogForm.value)}
                >
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
