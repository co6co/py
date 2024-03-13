<template>
  <el-container>
    <el-main class="ui">
      <el-card>
        <template #header>
          <div class="card-header">
            <h4>
              <el-icon><Bell /></el-icon> 告警详情
            </h4>
          </div>
        </template>
        <!--主内容-->
        <div style="width: 100%; padding: 0 5px; overflow: hidden">
          <img-video :viewOption="data_module.data"></img-video>
        </div>
        <template #footer>
          <el-descriptions title="详细信息" direction="vertical" :column="2" border>
            <el-descriptions-item label="告警站点">{{
              data_module.currentRow?.siteName
            }}</el-descriptions-item>
            <el-descriptions-item label="告警设备">{{
              data_module.currentRow?.boxName
            }}</el-descriptions-item>
            <el-descriptions-item label="告警类型">{{
              data_module.currentRow?.alarmType
            }}</el-descriptions-item>
            <el-descriptions-item label="告警描述" :span="2">{{
              data_module.currentRow?.alarmTypeDesc
            }}</el-descriptions-item>
            <el-descriptions-item label="告警时间" :span="2">{{
              data_module.currentRow?.alarmTime
            }}</el-descriptions-item>
          </el-descriptions>
        </template>
      </el-card>
    </el-main>
  </el-container>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive } from 'vue'
import {
  ElMessage,
  ElMessageBox,
  type FormRules,
  type FormInstance,
  ElTreeSelect,
  dayjs
} from 'element-plus'
import {
  Delete,
  Edit,
  Search,
  Compass,
  MoreFilled,
  Download,
  ArrowUp,
  ArrowDown
} from '@element-plus/icons-vue'
import * as api from '../api/alarm'
import { imgVideo, types } from '../components/player'
import { showLoading, closeLoading } from '../components/Logining'
import { type AlarmItem, getResources } from '../components/biz'
import { getQueryVariable } from '../utils'

interface AlertCategory {
  alarmType: string
  desc: string
}
interface Data_Module {
  currentRow?: AlarmItem
  categoryList: AlertCategory[]
  data: Array<types.resourceOption>
}
const data_module = reactive<Data_Module>({ categoryList: [], data: [] })
// 获取表格数据
const getData = () => {
  showLoading()
  let uid = getQueryVariable('uid')
  if (!uid)uid= getQueryVariable('id')
  if (!uid ) {
    ElMessage.error('获取查询参数失败！')
    return
  }
  api
    .get_one(uid)
    .then((res) => {
      data_module.currentRow = res.data
      data_module.data = getResources(res.data)
    })
    .finally(() => {
      closeLoading()
    })
}

const getAlarmCategory = async () => {
  const res = await api.alert_category_svc()
  data_module.categoryList = res.data
  getData()
}
getAlarmCategory()
</script>
<style scoped lang="less"></style>
