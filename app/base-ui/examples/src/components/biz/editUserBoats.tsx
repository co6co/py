/**
 * 修改用户与船的关联
 * 可批量操作
 *  *
 */
import { ref, reactive, defineComponent, defineExpose, nextTick } from 'vue'
import {
  ElDialog,
  ElForm,
  ElFormItem,
  ElSelect,
  ElOption,
  ElMessage,
  ElButton,
  ElTreeV2,
  type FormInstance,
  type FormRules,
  ElEmpty,
  ElRow,
  ElCol,
  type TreeNode,
  ElText,
  ElCard,
  ElCheckbox
} from 'element-plus'
import {
  type TreeNodeData,
  type TreeData,
  type TreeKey
} from 'element-plus/es/components/tree/src/tree.type'
import { User } from '@element-plus/icons-vue'
import * as api from '../../api/boat'
import user_api from '../../api/sys/user'
import * as group_api from '../../api/group'
import * as types from 'co6co'
import { minus, traverseTreeData } from 'co6co'
import '../../assets/css/c/editUserBoats.css'
import { showLoading, closeLoading } from '../../components/Logining'
import { tree_props } from '../../hook/useTreeProps'
import useNotifyAudit, { NotifyType } from '../../hook/useNotifyAudit'

export interface FromData {
  userId: number
  add: number[]
  remove: number[]
}

export interface dialogDataType {
  visible: boolean
  title?: string
  id: number
  loading: boolean
  treeDefaultChecked: Array<number>
  treeAllChecked: Array<number>
  fromData: FromData
}

export default defineComponent({
  name: 'EditUserBoat',
  emits: {
    saved: (userId: number) => true
  },
  methods: {
    onOpenDialog: () => {} //todo debug
  },
  setup(_, context) {
    const dialogForm = ref<FormInstance>()
    var ruleUserId = (
      rule: any,
      value: any,
      callback: (error?: string | Error | undefined) => void
    ) => {
      const reg = /^\d+$/
      if (value === '' || value === undefined || value == null) {
        callback(new Error('请选择关联用户'))
      } else {
        if (!reg.test(value)) {
          callback(new Error('请选择关联用户'))
        } else {
          callback()
        }
      }
    }
    var ruleboatIds = (
      rule: any,
      value: any,
      callback: (error?: string | Error | undefined) => void
    ) => {
      if (
        (form.fromData.add && form.fromData.add.length > 0) ||
        (form.fromData.remove && form.fromData.remove.length > 0)
      ) {
        callback()
      } else {
        callback(new Error('关联的船未改变,不需要保存！'))
      }
    }
    const rules: FormRules = {
      userId: [
        { required: true, validator: ruleUserId, message: '请选择关联用户', trigger: ['change'] }
      ],
      boatIds: [{ required: true, validator: ruleboatIds, trigger: ['blur', 'change'] }]
    }

    const form = reactive<dialogDataType>({
      visible: false,
      id: 0,
      loading: false,
      treeDefaultChecked: [],
      treeAllChecked: [],
      fromData: {
        userId: -1,
        add: [],
        remove: []
      }
    })

    //其他api 操作
    type UserData = types.ISelect & { children: UserData[] }
    const userTreeData = ref<UserData[]>([])
    const getUserTree = async () => {
      const res = await user_api.get_select_svc()
      if (res.code == 0) {
        userTreeData.value = res.data.map((m) => {
          return { id: m.id, name: m.name, children: [] }
        })
      }
    }
    getUserTree()
    const boatElRef = ref<InstanceType<typeof ElTreeV2>>()

    type BoatGroupData = types.ITreeSelect & { groupType: string; userId: number | null }
    const boatTreeData = ref<BoatGroupData[]>([])
    const getboatSelect = async (userId: number) => {
      showLoading()
      form.treeDefaultChecked = []
      boatTreeData.value = []
      form.treeAllChecked = []
      group_api
        .get_tree_svc(userId)
        .then((res) => {
          if (res.code == 0) {
            boatTreeData.value = res.data
            let isAllchecked = true
            traverseTreeData(res.data, (d) => {
              const dt = d as BoatGroupData
              if (dt.groupType == 'site') {
                if (dt.userId == userId) form.treeDefaultChecked.push(d.id)
                else isAllchecked = false
                form.treeAllChecked.push(d.id)
              }
            })
            //解决因第一项选中，后面加载的后仍被选中问题
            boatElRef.value?.setCheckedKeys(form.treeDefaultChecked)
            allChecked.value = isAllchecked
          }
        })
        .finally(() => {
          closeLoading()
        })
    }

    /*
    const traverseTreeData= (tree:Array<BoatGroupData>, func:(data:BoatGroupData)=>void) =>{
      tree.forEach((data) => { 
        data.children && traverseTreeData(data.children, func) // 遍历子树
        func(data)
      })
    } 
    */

    const onUserNodeClick = (data: TreeNodeData, node: TreeNode, e: MouseEvent) => {
      form.fromData.userId = data.id
      getboatSelect(data.id)
    }
    //是否被全选
    const checkAllChecked = () => {
      let nodes = boatElRef.value?.getCheckedNodes()
      if (nodes) {
        const siteNode = nodes.filter((m) => m.groupType == 'site').map((m) => m.id)
        if (siteNode.length == form.treeAllChecked.length) allChecked.value = true
        else allChecked.value = false
      }
    }
    //选择发生改变
    const onCheck = (data: TreeNodeData, checked: boolean) => {
      nextTick(() => {
        checkAllChecked()
      })
    }
    //end
    const allChecked = ref(false)
    const onAllCheck = () => {
      if (allChecked.value) {
        let checked: Array<number> = []
        traverseTreeData(boatTreeData.value, (d) => {
          const dt = d as BoatGroupData
          if (dt.groupType == 'site') {
            checked.push(d.id)
          }
        })
        boatElRef.value?.setCheckedKeys(checked)
      } else {
        boatElRef.value?.setCheckedKeys([])
      }
    }
    const onOpenDialog = () => {
      form.visible = true
      form.id = -1
      form.title = '设置用户关联的船只'
      form.fromData.userId = -1
      form.fromData.add = []
      form.fromData.remove = []
      boatTreeData.value = []
      allChecked.value = false
    }
    const { notifyAuditSystem } = useNotifyAudit()
    const onNotifyChanged = (userId: number) => {
      notifyAuditSystem({
        type: NotifyType.related_boat_changed,
        state: true,
        failMessage: '用户关联船通知失败',
        userId: userId
      })
    }
    const save = (formEl: FormInstance | undefined) => {
      let nodes = boatElRef.value?.getCheckedNodes()
      if (nodes) {
        const siteNode = nodes.filter((m) => m.groupType == 'site').map((m) => m.id)
        form.fromData.add = minus(siteNode, form.treeDefaultChecked) as number[]
        form.fromData.remove = minus(form.treeDefaultChecked, siteNode) as number[]
      }
      if (!formEl) return
      formEl.validate((value) => {
        if (value) {
          form.loading = true
          api
            .edits_svc(form.fromData)
            .then((res) => {
              if (res.code == 0) {
                form.visible = false
                ElMessage.success(`保存成功`)
                onNotifyChanged(form.fromData.userId)
                context.emit('saved', form.fromData.userId)
              } else {
                ElMessage.error(`保存失败:${res.message}`)
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
    context.expose({
      onOpenDialog
    })

    const dialogSlots = {
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
    }

    const leftCardSlots = {
      header: () => {
        return <ElFormItem label="请选择关联用户" labelWidth={135} prop="userId"></ElFormItem>
      }
    }
    const rightCardSlots = {
      header: () => {
        return <ElFormItem label="请选择关联船只" labelWidth={135} prop="boatIds"></ElFormItem>
      }
    }
    return () => {
      //可以写某些代码
      return (
        <>
          <ElDialog
            title={form.title}
            v-model={form.visible}
            v-slots={dialogSlots}
            style="with:90%; height:80%"
          >
            <ElForm labelWidth={90} ref={dialogForm} rules={rules} model={form.fromData}>
              <ElRow>
                <ElCol span={9}>
                  <ElCard v-slots={leftCardSlots}>
                    <ElTreeV2
                      icon={User}
                      props={tree_props}
                      data={userTreeData.value}
                      onNode-click={onUserNodeClick}
                    ></ElTreeV2>
                  </ElCard>
                </ElCol>
                <ElCol span={1}></ElCol>
                <ElCol span={14}>
                  <ElCard v-slots={rightCardSlots}>
                    {boatTreeData.value && boatTreeData.value.length > 0 ? (
                      <>
                        <ElCheckbox
                          v-model={allChecked.value}
                          onChange={onAllCheck}
                          style="margin-left: 23px;"
                        >
                          全选
                        </ElCheckbox>
                        <ElTreeV2
                          showCheckbox={true}
                          onCheck-change={onCheck}
                          ref={boatElRef}
                          defaultCheckedKeys={form.treeDefaultChecked}
                          props={tree_props}
                          data={boatTreeData.value}
                        ></ElTreeV2>
                      </>
                    ) : (
                      <ElEmpty description="未加载数据" />
                    )}
                  </ElCard>
                </ElCol>
              </ElRow>
            </ElForm>
          </ElDialog>
        </>
      )
    }
  }
})
