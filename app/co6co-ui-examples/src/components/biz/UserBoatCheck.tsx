import { defineComponent, reactive, ref, type PropType } from 'vue'
import * as api from '../../api/boat'
import {
  ElButtonGroup,
  ElEmpty,
  ElMessage,
  ElButton,
  ElDialog,
  ElRow,
  ElCol,
  ElCard,
  ElTable,
  ElTableColumn,
  ElScrollbar
} from 'element-plus'
import { Timer, Location, Bell } from '@element-plus/icons-vue'
import { showLoading, closeLoading } from 'co6co'

interface userListItem {
  id: number
  userName: string
}
interface boatListItem {
  id: number
  name: string
}

export interface dialogDataType {
  visible: boolean
  title?: string
  loading: boolean
  userListData: userListItem[]
  boatListData: boatListItem[]
}

export default defineComponent({
  name: 'UserBoatCheck',
  methods: {
    onOpenDialog: () => {} //todo debug
  },
  setup(props, context) {
    const getData = async () => {
      showLoading()
      api
        .check_svc()
        .then((res) => {
          if (res.code == 0) {
            form.userListData = res.data.userList
            form.boatListData = res.data.boatList
          }
        })
        .catch((e) => {
          ElMessage.error(`加载数据失败，请刷新重试！${e.message}`)
        })
        .finally(() => {
          closeLoading()
        })
    }

    const form = reactive<dialogDataType>({
      visible: false,
      loading: false,
      userListData: [],
      boatListData: []
    })

    const onOpenDialog = () => {
      getData()
      form.visible = true
      form.title = '未关联的用户及船信息'
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
          </span>
        )
      }
    }
    const leftCardSlots = {
      header: () => {
        return '未关联的用户'
      }
    }
    const rightCardSlots = {
      header: () => {
        return '未关联的船'
      }
    }
    return () => {
      //可以写某些代码
      return (
        <>
          <ElDialog title={form.title} v-model={form.visible} v-slots={dialogSlots} width={'80%'}>
            <ElRow style={'height:56vh'}>
              <ElCol span={12}>
                <ElCard v-slots={leftCardSlots}>
                  <ElTable
                    height={'48vh'}
                    highlightCurrentRow={true}
                    data={form.userListData}
                    rowClassName={'tableRowProp'}
                    border={true}
                    class={'table'}
                    headerCellClassName={'table-header'}
                  >
                    <ElTableColumn label="ID" prop="id" width={120}></ElTableColumn>
                    <ElTableColumn label="用户名" prop="userName"></ElTableColumn>
                  </ElTable>
                </ElCard>
              </ElCol>
              <ElCol span={12}>
                <ElCard v-slots={rightCardSlots}>
                  <ElTable
                    height={'48vh'}
                    highlightCurrentRow={true}
                    data={form.boatListData}
                    rowClassName={'tableRowProp'}
                    border={true}
                    class={'table'}
                    headerCellClassName={'table-header'}
                  >
                    <ElTableColumn label="ID" prop="id" width={180}></ElTableColumn>
                    <ElTableColumn label="船名" prop="name"></ElTableColumn>
                  </ElTable>
                </ElCard>
              </ElCol>
            </ElRow>
          </ElDialog>
        </>
      )
    }
  }
})
