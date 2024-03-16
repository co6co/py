<template>
  <el-card class="box-card">
    <!--header-->
    <template #header>
      <div class="card-header">
        <el-input v-model="tree_module.query.name" placeholder="点位名称">
          <template #append>
            <el-button :icon="Search" @click="tree_module.onSearch" />
          </template>
        </el-input>
      </div>
    </template>
    <!--content-->
    <el-scrollbar>
      <div class="content">
        <el-tree
          v-if="hasData"
          highlight-current
          @node-click="onNodeCheck"
          ref="treeRef"
          class="filter-tree"
          :data="tree_module.data"
          :props="tree_module.defaultProps"
          default-expand-all
          :filter-node-method="tree_module.filterNode"
        >
          <template #default="{ node, data }">
            <span>
              <!-- 没有子级所展示的图标 -->
              <i v-if="data.devices"
                ><el-icon> <Avatar /> </el-icon
              ></i>
              <i v-else>
                <el-tooltip :content="types.DeviceState[data.state]">
                  <el-icon :class="[{ 'is-loading': data.state == 0 }, 'state_' + data.state]">
                    <component :is="data.statueComponent" />
                  </el-icon>
                </el-tooltip>
              </i>
              <span class="label">
                <el-tooltip :content="data.deviceDesc || node.label">
                  <el-text truncated>{{ node.label }} </el-text>
                </el-tooltip>
              </span>
            </span>
          </template>
        </el-tree>
        <el-empty v-else></el-empty>
      </div>
    </el-scrollbar>
    <!--footer-->
    <template #footer>
      <div class="context">
        <el-pagination
          v-if="hasData"
          background
          layout="prev,next"
          :total="tree_module.total"
          :current-page="tree_module.query.pageIndex"
          :page-size="tree_module.query.pageSize"
          @current-change="tree_module.pageChange"
        />
      </div>
    </template>
  </el-card>
</template>
<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, markRaw } from 'vue'
import {
  ElMessage,
  ElMessageBox, 
  type FormInstance,
  ElTreeSelect,
  ElTree
} from 'element-plus'
import { type TreeNode, type TreeOptionProps } from 'element-plus/es/components/tree-v2/src/types'
import { type TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import {
  Delete,
  Edit,
  Search,
  Compass,
  MoreFilled,
  Download,
  CloseBold,
  VideoCamera,
  Avatar,
  ArrowUp,
  ArrowDown,
  Loading,
  Link,
  Connection,
  Refresh
} from '@element-plus/icons-vue'
import * as api from '../../..//api/site'
import * as gb_api from '../../..//api/deviceState'
import { showLoading, closeLoading } from '../../../components/Logining'
import * as types from './types'
import { GbDeviceState } from '../gb28181' 
import { Storage, SessionKey } from '../../../store/Storage'
import {GreenVideoCamera,RedVideoCamera} from '../../../components/icons/screenIcon'
let storeage = new Storage()

interface Emits {
  (e: 'nodeClick', box?: types.BoxDevice, device?: types.DeviceData, streams?: types.Stream[]): void
}
const emits = defineEmits<Emits>()
const treeRef = ref<InstanceType<typeof ElTree>>()
interface Tree {
  [key: string]: any
}
interface Query extends IpageParam {
  name: string
}

interface tree_module {
  query: Query
  data: Array<types.Site>
  currentItem?: types.Site
  currentDevice?: types.DeviceData
  total: number
  defaultProps: TreeOptionProps // { children: String; label:(treeData:any,treeNode:any)=>string };
  filterNode: (value: string, data: Tree) => boolean
  pageChange: (val: number) => void
  onSearch: () => void
}
const tree_module = reactive<tree_module>({
  query: {
    name: '',
    pageIndex: 1,
    pageSize: 20,
    order: 'asc',
    orderBy: ''
  },
  data: [],
  total: 0,
  filterNode: (value: string, data: Tree) => {
    if (!value) return true
    return data.label.includes(value)
  },
  // 分页导航
  pageChange: (val: number) => {
    tree_module.query.pageIndex = val
    getData()
  },
  onSearch: () => {
    getData()
  },
  defaultProps: {
    children: 'devices',
    label: 'name'
  }
})
let sessionId = storeage.get(SessionKey)
// 获取表格数据
const getData = () => {
  showLoading()
  api
    .list_svc(tree_module.query)
    .then((res) => {
      if (res.code == 0) {
        for (let i = 0; i < res.data.length; i++) {
          let devictState = types.DeviceState.loading
          //1. 如果 监控球机 0，删除
          if (res.data[i].devices && res.data[i].devices.length == 0) {
            setStatueComponent(res.data[i], devictState)
            delete res.data[i].devices
          } //2. 如果 监控球机 1，移动值 为 res.data[i] 属性
          else if (res.data[i].devices && res.data[i].devices.length == 1) {
            res.data[i].device = res.data[i].devices[0]
            setStatueComponent(res.data[i], devictState)
            delete res.data[i].devices
          } else {
            //3. 多个监控球机 :多
            for (let j = 0; j < res.data[i].devices.length; j++) {
              setStatueComponent(res.data[i].devices[j], devictState)
            }
          }
        }
        tree_module.data = res.data
        tree_module.total = res.total || -1
        onSyncState()
      } else {
        ElMessage.error(res.message)
      }
    })
    .finally(() => {
      closeLoading()
    })
}
const hasData = computed(() => tree_module.data.length > 0)
const onNodeCheck = (row: types.Site | types.DeviceData, b: any, c: any, d: any) => { 
  if (row) {
    let site = row as types.Site
    if (site.device || site.devices) {
      let data = row as types.Site
      tree_module.currentItem = data
      //只有一个设备的点
      if (data.device) {
        let device = data.device
        tree_module.currentDevice = device
        let stream = device.streams 
        emits('nodeClick', data.box, device, stream)
      }
      //有多个设备的点 ，仅展开
      else if (data.devices) console.info('展开')
    } else if ((row as types.DeviceData).streams) {
      let device = row as types.DeviceData
      //点位
      tree_module.currentDevice = device

      emits('nodeClick', b.parent.data.box, device, device.streams)
    } else {
      emits('nodeClick', undefined, undefined)
    }
  } else {
    console.info('row is None')
  }
}
// 状态
const comMap = reactive([markRaw(Loading), markRaw(VideoCamera),markRaw(GreenVideoCamera),markRaw(RedVideoCamera)])
const setStatueComponent = (data: types.Site | types.DeviceData, state: types.DeviceState) => {
  data.state = state
  switch (state) {
    case types.DeviceState.loading:
      data.statueComponent = comMap[0]
      break
    case types.DeviceState.Connected:
      data.statueComponent = comMap[2]
      break
    default:
      data.statueComponent = comMap[3]
      break
  }
}

//查询通道是否在线
const queryChannel = (stateArray: gb_api.gbDeviceState[], channel_sip: string): boolean => {
  let result: gb_api.gbDeviceState[] = stateArray.filter((m) => m.uri.includes(channel_sip))
  return result.length > 0
}

const syncChannelState = (
  devices: types.DeviceData | types.DeviceData[],
  stateArray: gb_api.gbDeviceState[]
) => {
  if (Array.isArray(devices)) {
    for (let i = 0; i < devices.length; i++) {
      syncChannelState(devices[i], stateArray)
    }
  } else if (devices.streams && devices.streams.length > 0) {
    for (let i = 0; i < devices.streams.length; i++) {
      let channelKey = `channel${i + 1}_sip`
      let chanelSip = devices[channelKey]
      if (chanelSip && typeof chanelSip == 'string') {
        let exist = queryChannel(stateArray, chanelSip)
        devices.streams[i].valid = exist
        if (!devices.streams[i].url.includes('userid=') && sessionId)
          devices.streams[i].url = devices.streams[i].url + '&userid=' + sessionId
      } else {
        console.warn('未与sip通道对应的视频流可用状态为:true')
        devices.streams[i].valid = true
      }
    }
  }
}
const getAiAndCameraOnlineState = (aiExist: boolean, cameraExist: boolean) => {
  let result = types.DeviceState.Disconected
  if (aiExist) {
    result = cameraExist ? types.DeviceState.Connected : types.DeviceState.AIConnected
  } else {
    result = cameraExist ? types.DeviceState.CameraConnected : types.DeviceState.Disconected
  }
  return result
}
/**
 * 在线状态
 * 1. sip 在线  sip 地址实在 盒子上还是在相机上  现在认为在相机上
 * 2. 通道sip是否在线
 */
const onSynDeviceState = (stateArray: gb_api.gbDeviceState[]) => {
  for (let i = 0; i < tree_module.data.length; i++) {
    let site = tree_module.data[i]
    let aiExist = false
    //if (site.box) aiExist = queryChannel(stateArray, site.box.sip);
    //else aiExist=false;
    if (site.devices) {
      for (let j = 0; j < site.devices.length; j++) {
        if (j == 0) {
          aiExist = queryChannel(stateArray, site.devices[j].sip)
        }
        site.devices[j].channel1_sip
        let exist = queryChannel(stateArray, site.devices[j].channel1_sip)
        setStatueComponent(site.devices[j], getAiAndCameraOnlineState(aiExist, exist))
      }
      syncChannelState(site.devices, stateArray)
    } else if (site.device) {
      aiExist = queryChannel(stateArray, site.device.sip)
      let exist = queryChannel(stateArray, site.device.channel1_sip)
      setStatueComponent(site, getAiAndCameraOnlineState(aiExist, exist))
      syncChannelState(site.device, stateArray)
    } else {
      setStatueComponent(site, getAiAndCameraOnlineState(aiExist, false))
    }
  }
}
const onSyncState = () => {
  GbDeviceState(onSynDeviceState)
}
onMounted(() => {
  getData()
})
</script>
<style lang="less" scoped>
::v-deep .el-card__body {
  padding-top: 5px;
  height: 100%;
  .content {
    padding: 0;
    .el-tree-node__content {
      margin-left: -24px;
    }
    .label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 0 5px;
    }
  }
}

::v-deep .el-tree--highlight-current {
  .el-tree-node {
    &.is-current {
      .el-tree-node__content {
        background: #a1b9d5;
      }
    }
  }
}
.state_0 {
  color: #b3b02f;
}
.state_1 {
  color: green;
}
.state_2 {
  color: red;
}
.state_3 {
  //AIConnected
  color: rgb(102, 177, 255);
}
.state_4 {
  //CameraConnected
  color: rgb(83, 168, 255);
}
</style>
