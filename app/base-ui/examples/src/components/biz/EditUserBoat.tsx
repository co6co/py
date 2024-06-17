/**
 * 修改用户与船的关联 单个操作
 *
 */
import { ref, reactive, defineComponent, defineExpose } from 'vue'

import {
  ElDialog,
  ElForm,
  ElFormItem,
  ElSelect,
  ElOption,
  ElMessage,
  ElButton,
  type FormInstance,
  type FormRules
} from 'element-plus'
import * as api from '../../api/boat'
import * as group_api from '../../api/group'
import * as types from 'co6co'
import useUserSelect from '../../hook/useUserSelect'

export interface Item {
  id: number
  userId?: number
  boatId: string
  userName: string
  boatName: string
}

//Omit、Pick、Partial、Required
type FromData = Omit<Item, 'id' | 'userName' | 'boatName'>
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
    saved: () => true
  },
  methods: {
    onOpenDialog: (operation: types.Operation.Add | types.Operation.Edit, item?: Item) => {
      alert('ts demo..')
    }
  },
  setup(_, context) {
    const dialogForm = ref<FormInstance>()
    const rules: FormRules = {
      userId: [{ required: true, message: '请选择关联用户', trigger: ['blur'] }],
      boatId: [{ required: true, message: '请选择关联船', trigger: 'blur' }]
    }

    const form = reactive<dialogDataType>({
      visible: false,
      operation: types.Operation.Add,
      id: 0,
      loading: false,
      fromData: {
        boatId: ''
      }
    })

    //其他api 操作
    const { userSelect } = useUserSelect()
    const boatSelect = ref<types.ISelect[]>([])
    const getboatSelect = async () => {
      const res = await group_api.get_select_svc()
      if (res.code == 0) {
        boatSelect.value = res.data
      }
    }
    getboatSelect()
    //end

    const onOpenDialog = (operation: types.Operation.Add | types.Operation.Edit, item?: Item) => {
      form.visible = true
      form.operation = operation
      form.id = -1
      switch (operation) {
        case types.Operation.Add:
          form.title = '增加'
          form.fromData.userId = undefined
          form.fromData.boatId = ''
          break
        case types.Operation.Edit:
          if (item && item.id) {
            form.id = item.id
            form.title = '编辑'
            form.fromData.userId = item.userId
            form.fromData.boatId = item.boatId.toString()
          }
          break
      }
    }
    const save = (formEl: FormInstance | undefined) => {
      if (!formEl) return
      formEl.validate((value) => {
        if (value) {
          form.loading = true
          console.info(form.fromData)
          if (form.operation == types.Operation.Add) {
            api
              .add_svc(form.fromData)
              .then((res) => {
                if (res.code == 0) {
                  form.visible = false
                  ElMessage.success(`增加成功`)
                  context.emit('saved')
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
                  context.emit('saved')
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
            <ElForm
              labelWidth={90}
              ref={dialogForm}
              rules={rules}
              model={form.fromData}
              style="max-width: 460px"
            >
              <ElFormItem label="关联用户" prop="userId">
                <ElSelect
                  style="width: 160px"
                  clearable={true}
                  v-model={form.fromData.userId}
                  placeholder="请选择"
                >
                  <ElOption label="请选择" value=""></ElOption>
                  {userSelect.value.map((item: types.ISelect, index: number) => {
                    return <ElOption key={index} label={item.name} value={item.id}></ElOption>
                  })}
                </ElSelect>
              </ElFormItem>

              <ElFormItem label="关联船" prop="boatId">
                <ElSelect
                  style="width: 160px"
                  clearable={true}
                  v-model={form.fromData.boatId}
                  placeholder="请选择"
                >
                  {boatSelect.value.map((item: types.ISelect, index: number) => {
                    return (
                      <ElOption key={index} label={item.name} value={item.id.toString()}></ElOption>
                    )
                  })}
                </ElSelect>
              </ElFormItem>
            </ElForm>
          </ElDialog>
        </>
      )
    }
  }
})
