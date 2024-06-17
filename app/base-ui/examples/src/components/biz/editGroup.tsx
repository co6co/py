import { ref, reactive, defineComponent } from 'vue'

import {
  ElDialog,
  ElForm,
  ElFormItem,
  ElSelect,
  ElOption,
  ElInput,
  ElMessage,
  ElButton,
  type FormInstance,
  type FormRules
} from 'element-plus'
import * as api from '../../api/group'
import * as types from 'co6co'

export interface Item {
  id?: number
  boartName: string
  name: string
  ipCameraSerial: string
  boatPosNumber: string
  boatSerial: string
  parentId: number
}

export interface dialogDataType {
  visible: boolean
  operation: types.Operation
  title?: string
  id: number
  loading: boolean
  fromData: Item
}

export interface GroupStatus {
  group: Array<optionItem>
  postion: Array<optionItem>
  allowSetNumberGroup: Array<string>
}
export default defineComponent({
  name: 'EditGroupView',
  emits: {
    saved: () => true
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
        boartName: '',
        name: '',
        ipCameraSerial: '',
        boatPosNumber: '',
        boatSerial: '',
        parentId: 0
      }
    })

    //其他api 操作
    const groupState = ref<GroupStatus>()
    const getGroupState = async () => {
      const res = await api.get_status_svc()
      if (res.code == 0) {
        groupState.value = res.data
      }
    }
    getGroupState()
    //end
    const onOpenDialog = (operation: types.Operation.Add | types.Operation.Edit, item?: Item) => {
      form.visible = true
      form.operation = operation
      form.id = -1
      switch (operation) {
        case types.Operation.Add:
          form.title = '增加'
          ;(form.fromData.boartName = ''),
            (form.fromData.name = ''),
            (form.fromData.ipCameraSerial = ''),
            (form.fromData.boatPosNumber = ''),
            (form.fromData.boatSerial = '')
          break
        case types.Operation.Edit:
          if (item && item.id) {
            form.id = item.id
            form.title = '编辑'

            form.fromData.name = item.name
            form.fromData.ipCameraSerial = item.ipCameraSerial
            form.fromData.boatPosNumber = item.boatPosNumber
            form.fromData.boartName = item.boartName
            api.get_one_svc(item.parentId).then((res) => {
              form.fromData.boartName = res.data.name
              form.fromData.boatSerial = res.data.boatSerial
            })
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
            form.loading = false
          } else {
            api
              .update_svc(form.id, form.fromData)
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
              labelWidth={70}
              ref={dialogForm}
              rules={rules}
              model={form.fromData}
              style="max-width: 460px"
            >
              <ElFormItem label="船名" prop="boartName">
                <ElInput
                  readonly={true}
                  v-model={form.fromData.boartName}
                  placeholder="船名"
                ></ElInput>
              </ElFormItem>

              <ElFormItem label="船序列号" prop="boatSerial">
                <ElInput
                  readonly={true}
                  v-model={form.fromData.boatSerial}
                  placeholder="船序列号"
                ></ElInput>
              </ElFormItem>

              <ElFormItem label="设备列号" prop="ipCameraSerial">
                <ElInput
                  readonly={true}
                  v-model={form.fromData.ipCameraSerial}
                  placeholder="设备列号"
                ></ElInput>
              </ElFormItem>

              <ElFormItem label="设备名" prop="name">
                <ElInput
                  readonly={true}
                  v-model={form.fromData.name}
                  placeholder="设备名"
                ></ElInput>
              </ElFormItem>

              <ElFormItem label="部位编号" prop="name">
                <ElInput
                  v-model={form.fromData.boatPosNumber}
                  placeholder="zhc_jh36_jb[船类型_船编号_部位名称]"
                ></ElInput>
              </ElFormItem>
            </ElForm>
          </ElDialog>
        </>
      )
    }
  }
})
