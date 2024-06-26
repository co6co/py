<template>
  <div class="container-layout">
    <el-container>
      <el-header>
        <div class="handle-box">
          <el-select style="width: 160px" class="mr10" v-model="query.groupType" placeholder="类型">
            <el-option
              v-for="item in state_store.group"
              :key="item.key"
              :label="item.label"
              :value="item.key"
            />
          </el-select>
          <el-input
            style="width: 160px"
            v-model="query.name"
            placeholder="名称"
            class="handle-input mr10"
          ></el-input>
          <el-input
            style="width: 160px"
            v-model="query.boatSerial"
            placeholder="序列号"
            class="handle-input mr10"
          ></el-input>

          <el-input
            style="width: 160px"
            v-model="query.boatPosNumber"
            placeholder="部位编号"
            class="handle-input mr10"
          ></el-input>

          <el-input
            style="width: 160px"
            v-model="query.priority"
            placeholder="优先级"
            class="handle-input mr10"
          ></el-input>
          <el-button-group>
            <el-button :icon="Search" @click="onSearch">查询</el-button>
            <!-- <el-button type="primary" :icon="Sunny" @click="onResetQuery">重置</el-button>-->
            <el-button
              type="primary"
              v-permiss="getPermissKey(ViewFeature.push)"
              :icon="Top"
              @click="onOpenAssDiaglog"
              >自动推送</el-button
            >
            <el-button
              type="danger"
              v-permiss="getPermissKey(ViewFeature.reset)"
              :icon="Setting"
              @click="onReset"
              >重置优先级</el-button
            >

            <el-button
              type="warning"
              :icon="Connection"
              v-permiss="getPermissKey(ViewFeature.effective)"
              :disabled="!priorityNotice.priorityChanged"
              @click="onPriorityChanged"
              >优先级立即生效</el-button
            >
          </el-button-group>
        </div>
      </el-header>
      <el-container>
        <el-aside width="25%" style="height: 100%">
          <el-scrollbar>
            <el-tree
              ref="treeRef"
              style="max-width: 600px"
              :data="selectTreeModule.data"
              default-expanded-keys="[0]"
              node-key="id"
              @node-click="onTreeNodeClick"
              accordion
              highlight-current
              :props="selectTreeModule.defaultProps"
            />
          </el-scrollbar>
        </el-aside>
        <el-container>
          <el-main>
            <el-scrollbar>
              <el-table
                :data="tableData"
                @sort-change="onColChange"
                border
                class="table"
                header-cell-class-name="table-header"
                row-key="id"
                :tree-props="{ children: 'children' }"
              >
                <!-- <el-table-column type="selection" width="55" />-->

                <el-table-column label="序号" width="100" align="center">
                  <template #default="scope"> {{ scope.$index + 1 }} </template>
                </el-table-column>
                <el-table-column
                  prop="name"
                  label="名称"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                ></el-table-column>
                <el-table-column
                  width="110"
                  label="分组类型"
                  prop="groupType"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                >
                  <template #default="scope">
                    {{ scope.row.groupType }} -
                    {{ state_store.getGroupItem(scope.row.groupType)?.label }}
                  </template>
                </el-table-column>
                <el-table-column
                  width="100"
                  label="序列号"
                  prop="ipCameraSerial"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                >
                  <template #default="scope">
                    {{ scope.row.ipCameraSerial || scope.row.boatSerial }}
                  </template>
                </el-table-column>
                <el-table-column
                  width="120"
                  label="部位编号"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                  prop="boatPosNumber"
                ></el-table-column>

                <el-table-column
                  width="100"
                  label="优先级"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                  prop="priority"
                ></el-table-column>

                <el-table-column
                  width="100"
                  label="自动推送"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                  prop="autoPush"
                >
                  <template #default="scope">
                    {{ getAutoPushDesc(scope.row) }}
                  </template>
                </el-table-column>

                <el-table-column
                  width="120"
                  prop="updateTime"
                  label="更新时间"
                  sortable="custom"
                  :show-overflow-tooltip="true"
                ></el-table-column>
                <el-table-column label="操作" width="216" align="center">
                  <template #default="scope">
                    <el-button
                      text
                      v-permiss="getPermissKey(ViewFeature.settingNo)"
                      :icon="Setting"
                      v-if="state_store.allowSetting(scope.row.groupType)"
                      @click="onOpenEditDialog(scope.$index, scope.row)"
                    >
                      设置编号
                    </el-button>

                    <el-button
                      text
                      v-permiss="getPermissKey(ViewFeature.settingPriority)"
                      :icon="Setting"
                      v-if="state_store.allowSetPriority(scope.row.groupType)"
                      @click="onOpenSetGroupPriorityDialog(scope.$index, scope.row)"
                    >
                      设置船优先级
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-scrollbar>
          </el-main>
          <el-footer>
            <div class="pagination">
              <el-pagination
                background
                layout="total, prev, pager, next"
                :current-page="query.pageIndex"
                :page-size="query.pageSize"
                :total="pageTotal"
                @current-change="onPageChange"
              ></el-pagination>
            </div>
          </el-footer>
        </el-container>
      </el-container>
    </el-container>
    <edit-group ref="editViewRef" @saved="getData()"></edit-group>
    <set-group-priority ref="setGroupPriorityRef" @saved="onSavedPriority"></set-group-priority>
    <associated-diaglog
      ref="associatedDiaglogRef"
      @saved="onSaveAssociated"
      style="width: 40%; height: 80%; overflow: hidden"
      title="自动推送船选择"
    ></associated-diaglog>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue'
import { ElMessage, ElMessageBox, type FormRules, type FormInstance } from 'element-plus'
import {
  Delete,
  Sunny,
  Edit,
  Search,
  Compass,
  Top,
  Plus,
  Setting,
  Connection
} from '@element-plus/icons-vue'
import {
  get_tree_list_by_pid_svc,
  select_tree_svc,
  reset_boat_priority,
  association_service as ass_api
} from '../api/group'
import { group_state_store } from '../store/group'
import editGroup, { type Item } from '../components/biz/editGroup'
import setGroupPriority, {
  type Item as priorityItem,
  type FormItem
} from '../components/biz/setGroupPriority'
import { showLoading, closeLoading, Operation, type ITreeSelect, type IPageParam } from 'co6co'

import useNotifyAudit, { NotifyType } from '../hook/useNotifyAudit'
import associatedDiaglog from '../components/sys/associated'
import { usePermission, ViewFeature } from '../hook/sys/useRoute'
const { getPermissKey } = usePermission()

const state_store = group_state_store()
state_store.refesh().then((res) => {})
interface IQueryItem extends IPageParam {
  name?: string
  groupType?: string
  boatPosNumber?: string
  boatSerial?: string
  priority?: number
}
const query = reactive<IQueryItem>({
  groupType: 'site',
  pageIndex: 1,
  pageSize: 10,
  order: 'asc',
  orderBy: ''
})

interface ISelectTreeModule {
  currentId: number
  //contentGroupType:string,
  data: ITreeSelect[]
  defaultProps: { children: string; label: string }
}
const selectTreeModule = reactive<ISelectTreeModule>({
  currentId: 0,
  data: [],
  //contentGroupType:"site",
  defaultProps: {
    children: 'children',
    label: 'name'
  }
})

const getSelectTreeData = () => {
  showLoading()
  select_tree_svc()
    .then((res) => {
      selectTreeModule.data = res.data
    })
    .finally(() => {
      closeLoading()
    })
}

const onTreeNodeClick = (node: ITreeSelect, nodeObj: any, tree: any, ent: any) => {
  selectTreeModule.currentId = node.id
  getData()
}

const tableData = ref<any[]>([])
const pageTotal = ref(0)
// 获取表格数据
const getData = () => {
  showLoading()
  get_tree_list_by_pid_svc(selectTreeModule.currentId, query)
    .then((res) => {
      if (res.code == 0) {
        tableData.value = res.data
        pageTotal.value = res.total || -1
      } else {
        ElMessage.error(res.message)
      }
    })
    .finally(() => {
      closeLoading()
    })
}
getSelectTreeData()
getData()
// 查询操作
const onSearch = () => {
  getData()
}

const onColChange = (column: any) => {
  query.order = column.order === 'descending' ? 'desc' : 'asc'
  query.orderBy = column.prop
  if (column) getData() // 获取数据的方法
}
const onPageChange = (val: number) => {
  query.pageIndex = val
  getData()
}

const onReset = () => {
  ElMessageBox.confirm(`确定要重置所有船的优先级吗？`, '提示', {
    type: 'warning'
  })
    .then(() => {
      showLoading()
      reset_boat_priority()
        .then((res) => {
          if (res.code == 0) ElMessage.success(res.message), getData()
          else ElMessage.error(res.message)
        })
        .finally(() => {
          closeLoading()
        })
    })
    .catch(() => {})
}
const editViewRef = ref<typeof editGroup>()
const onOpenEditDialog = (index: number, row: Item) => {
  editViewRef.value?.onOpenDialog(row ? Operation.Edit : Operation.Add, row)
}

interface IPriorityNotice {
  priorityTempValue?: number
  priorityChanged: boolean
}
const priorityNotice = reactive<IPriorityNotice>({
  priorityChanged: false
})
const setGroupPriorityRef = ref<InstanceType<typeof setGroupPriority>>()
const onOpenSetGroupPriorityDialog = (index: number, row: priorityItem) => {
  priorityNotice.priorityTempValue = row.priority
  setGroupPriorityRef.value?.openDialog(row)
}
const onSavedPriority = (_: any, item: FormItem) => {
  if (priorityNotice.priorityTempValue != item.priority && !priorityNotice.priorityChanged) {
    priorityNotice.priorityChanged = true
  }
  getData()
}
//通知
const { notifyAuditSystem } = useNotifyAudit()
const onPriorityChanged = () => {
  notifyAuditSystem({
    type: NotifyType.boat_priority_changed,
    state: true,
    failMessage: '船优先级通知失败',
    message: '船优先级通知成功'
  })
}
//自动推送的船
const associatedDiaglogRef = ref<InstanceType<typeof associatedDiaglog>>()
const onOpenAssDiaglog = () => {
  associatedDiaglogRef.value?.openDialog(
    1,
    ass_api.get_association_svc,
    ass_api.save_association_svc,
    (m: any) => m.groupType == 'site'
  )
}

//onSaveAssociated
const onSaveAssociated = () => {
  notifyAuditSystem({
    type: NotifyType.auto_audit_boat_changed,
    state: true,
    failMessage: '设置自动推送通知失败',
    message: '设置自动推送通知成功'
  })
}

const getAutoPushDesc = (row?: { groupType: string; autoPush: number }) => {
  if (row && row.groupType == 'site') {
    if (row.autoPush == 1) return '推送'
    return '不推送'
  }
  return ''
}
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
</style>
