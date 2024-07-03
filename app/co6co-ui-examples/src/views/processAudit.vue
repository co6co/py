<template>
  <div class="container-layout" tabindex="-1" @keydown="onKeyDown">
    <el-container>
      <el-header>
        <div class="header">
          <el-row>
            <el-col :span="12" style="line-height: 32px">
              <el-text class="mx-1"
                >当前审核记录数{{ currentNum }}/{{ tableModule.recordTotal }},完成
                {{ (precentage * 100).toFixed(2) }}%</el-text
              >
              <el-tag :type="tagType.type">{{ tagType.desc }}</el-tag>
            </el-col>
            <el-col :span="12">
              <el-input
                style="width: 330px"
                @keydown.stop
                v-model.number="tableModule.query.record_num"
                placeholder="输入请求数量"
              >
                <template #prepend>请求数量</template>
                <template #append>
                  <el-button :icon="Pointer" @click="getData">接单</el-button>
                </template>
              </el-input>
              <!--心跳没有UI-->
              <heart-beat ref="heartBeatRef" :interval="5000" />
              <div style="float: right; padding: 0 10px">
                <el-text
                  >审核倒计时 <el-tag :type="timeType">{{ tableModule.timeout }}</el-tag
                  >秒
                </el-text>
              </div>
            </el-col>
          </el-row>
        </div>
      </el-header>
      <el-main>
        <!--主内容-->
        <auditview :record="tableModule.currentItem" @audited="onAudit">
          <template #default>
            <div class="navContent">
              <el-button-group class="ml-4">
                <el-button
                  type="primary"
                  @click="onNavClick(-1)"
                  :icon="ArrowLeft"
                  :disabled="!previousEnabled"
                  >上一条
                </el-button>
                <el-button type="primary" @click="onNavClick(1)" :disabled="!nextEnabled"
                  >下一条 <el-icon> <ArrowRight /></el-icon>
                </el-button>
                <el-input number style="width: 80px" v-model="currentNum" @keydown.stop>
                  <template #prefix>第</template>
                  <template #suffix> 条 </template>
                </el-input>
              </el-button-group>
              <el-button-group class="ml-8" style="float: right">
                <!--批量提交-->
                <el-button
                  v-if="appSetting.batchAudit"
                  type="danger"
                  :icon="Finished"
                  @click="onSubmit"
                  >提交</el-button
                >
              </el-button-group>
            </div>
          </template>
        </auditview>
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive, onMounted, computed, watch } from 'vue'
import { ElMessage, ElMessageBox, ElTag } from 'element-plus'
import { Pointer, ArrowLeft, ArrowRight, Finished } from '@element-plus/icons-vue'
import { format, showLoading, closeLoading } from 'co6co'
import * as api from '../api/pd'
// eslint-disable-next-line camelcase
import * as appApi from '../api/app'
import { IntervalTime as heartBeat } from 'co6co'

import auditview, {
  AuditType,
  getAuditTypeDesc,
  getTagType,
  type IAuditResult
} from '../components/process/auditview'
import useUserState from '../hook/useUserState'

type Query = api.Iorders_param
interface Table_module {
  query: Query
  jobId: bigint
  data: api.Iorders_record[]
  currentItem?: api.Iorders_record
  currentIndex: number //当前记录
  recordTotal: number //接收到的记录数
  isBatchSubmited: boolean //批量审核是否已提交
  timerIsRuning: boolean
  timeout: number //审核超时时间 s
}

const tableModule = reactive<Table_module>({
  query: {
    user_id: -1,
    record_num: 20
  },
  jobId: BigInt(0),
  data: [],
  currentIndex: 0,
  recordTotal: 0, //获得的记录数
  isBatchSubmited: true,
  timerIsRuning: false,
  timeout: 0
})
const onNavClick = (i: number) => {
  tableModule.currentIndex += i
  tableModule.currentItem = tableModule.data[tableModule.currentIndex]
}
const currentNum = computed({
  get() {
    return tableModule.currentIndex + 1
  },
  set(val: number) {
    if (tableModule.data.length == 0) tableModule.currentIndex = 0
    else if (val >= tableModule.data.length) tableModule.currentIndex = tableModule.data.length - 1
    else tableModule.currentIndex = val - 1
    tableModule.currentItem = tableModule.data[tableModule.currentIndex]
  }
})

const precentage = computed(() => {
  if (tableModule.recordTotal > 0) return currentNum.value / tableModule.recordTotal
  return 0.0
})
const previousEnabled = computed(() => {
  if (tableModule.recordTotal > 0 && tableModule.currentIndex != 0) return true
  return false
})
const nextEnabled = computed(() => {
  if (tableModule.recordTotal > 0 && tableModule.currentIndex + 1 != tableModule.recordTotal)
    return true
  return false
})
const tagType = ref<{ type: 'info' | 'warning' | 'success' | 'danger'; desc: string }>({
  type: 'info',
  desc: ''
})

watch(
  () => tableModule.currentItem?.manual_audit_result,
  (n?: number) => {
    if (typeof n == 'number') {
      tagType.value.desc = getAuditTypeDesc(n)
      tagType.value.type = getTagType(n)
    } else {
      tagType.value.desc = '未审核'
      tagType.value.type = 'danger'
    }
  },
  { immediate: true }
)
const { getCurrentUserId } = useUserState()

const timeType = computed(() => {
  if (tableModule.timerIsRuning) {
    if (tableModule.timeout > 0 && tableModule.timeout < 60) return 'warning'
    else return 'info'
  } else {
    return tableModule.timeout > 0 ? 'success' : 'danger'
  }
})

let timer: any = null
const startTime = () => {
  stopTime()
  tableModule.timerIsRuning = true
  timer = setInterval(() => {
    if (tableModule.timeout > 0) tableModule.timeout = tableModule.timeout - 1
    else {
      ElMessageBox.alert('未能在规定时间审核系统将回收分配的审核记录，提交数据将不记录！')
      stopTime()
    }
  }, 1000)
}
const stopTime = () => {
  if (timer) clearInterval(timer)
  tableModule.timerIsRuning = false
}

// 获取表格数据
const getData = () => {
  tableModule.query.user_id = getCurrentUserId()
  if (typeof tableModule.query.record_num != 'number') {
    ElMessage.error('输入请求数不正确')
    return false
  }
  if (!allowGetData()) {
    ElMessage.warning('当前任务未完成！')
    return false
  }
  showLoading()
  api
    .assign_orders(tableModule.query)
    .then((res) => {
      if (api.isSuccess(res)) {
        tableModule.currentIndex = 0
        tableModule.jobId = res.job_id
        tableModule.data = res.records
        if (tableModule.data.length > 0) {
          tableModule.isBatchSubmited = false
          tableModule.currentItem = tableModule.data[0]
          tableModule.recordTotal = res.records.length || 0
          tableModule.timeout = res.time_out * 60
          startTime()
        } else {
          ElMessage.warning(`数据正在准备中，请稍后重试 ;${res.message}`)
        }
      } else {
        ElMessage.error(res.message)
      }
    })
    .catch((err) => {
      console.info(err)
    })
    .finally(() => {
      closeLoading()
    })
}

const heartBeatRef = ref<typeof heartBeat>()
const getPromise = () => {
  const id = getCurrentUserId()
  const date = format(new Date(), 'YYYY-MM-DD HH:mm:ss')
  if (id) return api.heartbeat({ user_id: id, time: date })
  else return Promise.reject('未找到Session')
}
onMounted(() => {
  if (heartBeatRef.value) {
    heartBeatRef.value.runinng(getPromise, (data: { state: any }) => {
      //console.info('心跳返回', data.state)
    })
  }
})

//审核事件
const onAudit = (data: IAuditResult) => {
  if (tableModule.currentItem) {
    //设置该记录的审核结果
    tableModule.currentItem.manual_audit_result = data.type
    if (data.label) tableModule.currentItem.label = data.label
    //单条记录审核提交提交
    if (appSetting.value && !appSetting.value.batchAudit) {
      const param = getsubmitParam([tableModule.currentItem])
      _submit(param)
    }
  } else console.warn('当前记录未NULL!')
}

const allowGetData = () => {
  if (!tableModule.timerIsRuning) return true
  if (appSetting.value.batchAudit) {
    return tableModule.isBatchSubmited
  } else {
    let audited = isAudited()
    return audited
  }
}
//提交相关代码
const isAudited = () => {
  let result = tableModule.data.filter((m) => m.manual_audit_result == undefined)
  if (result.length == 0) return true
  else {
    const unAuditNumber: number[] = []
    for (const element of result) {
      const index = tableModule.data.findIndex((x) => x.id == element.id)
      unAuditNumber.push(index + 1)
    }
    ElMessage.error(`第${unAuditNumber.join(',')}未审核！`)
    return false
  }
}

const _submit = (data: api.Iaudit_param) => {
  showLoading()
  api
    .manual_audit(data)
    .then((res) => {
      if (api.isSuccess(res)) {
        if (res.message) ElMessage.success(res.message)
        else ElMessage.success('提交成功')
        stopTime()
      } else {
        ElMessage.success(`审核失败：${res.message}`)
      }
    })
    .catch((e) => {
      ElMessage.error(`审核异常：${e}`)
    })
    .finally(() => closeLoading())
}
//
const getsubmitParam = (data: api.Iorders_record[]) => {
  const userId = getCurrentUserId()
  let records = data.map((r) => ({
    job_id: tableModule.jobId, // 63754308716557023406, # 订单任务流水号
    id: r.id, // 7211, # 记录id
    manual_audit_result: r.manual_audit_result!, // 1, # 审核结果
    label: r.label,
    auditor_id: userId // 3 # 审核员id
  }))
  let result: api.Iaudit_param = { user_id: userId, records }

  return result
}
//批量保存
const onSubmit = () => {
  const result = isAudited()
  if (result) {
    const param = getsubmitParam(tableModule.data)
    _submit(param)
    tableModule.isBatchSubmited = true
  }
}

const onKeyDown = (e: KeyboardEvent) => {
  if (e.key == 'ArrowUp' && previousEnabled.value) onNavClick(-1)
  else if (e.key == 'ArrowDown' && nextEnabled.value) onNavClick(1)
  else if (['1', '2', '3', '4'].includes(e.key) && tableModule.currentItem) {
    onAudit({ type: Number(e.key) - 1 })
  }
  e.stopPropagation()
}

const appSetting = ref<appApi.ClientConfig>({ batchAudit: false })
appApi.get_config().then((res) => {
  appSetting.value = res.data
})
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
.container-layout {
  &:focus-visible,
  &:focus {
    outline: 0px !important;
  }
}
::v-deep .navContent {
  position: absolute;
  bottom: 0;
  right: 5px;
  .el-input__inner {
    text-align: center;
  }
}
.header {
  background-color: white;
  padding: 10px 5px;
}
.el-row {
  height: 100%;
}

.view .title {
  color: var(--el-text-color-regular);
  font-size: 18px;
  margin: 10px 0;
}

.view .value {
  color: var(--el-text-color-primary);
  font-size: 16px;
  margin: 10px 0;
}

::v-deep .view .radius {
  height: 40px;
  width: 70%;
  border: 1px solid var(--el-border-color);
  border-radius: 0;
  margin-top: 20px;
}

::v-deep .el-table tr,
.el-table__row {
  cursor: pointer;
}

.formItem {
  display: flex;
  align-items: center;
  display: inline-block;

  .label {
    display: inline-block;
    color: #aaa;
    padding: 0 5px;
  }
}

::v-deep .el-dialog__body {
  height: 70%;
  overflow: auto;
}

.menuInfo {
  .el-menu {
    width: auto;
    .el-menu-item {
      padding: 10px;
      height: 40px;
    }
  }
}
</style>
