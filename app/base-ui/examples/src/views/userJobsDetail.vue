<template>
  <div class="container-layout" tabindex="-1">
    <el-container>
      <el-header>
        <div class="header">
          <el-row>

            <el-col :span="12" style="line-height: 32px">
              <el-text>当前任务用户：  {{ getUserName(table_module.job?.userId)}} </el-text>
              <el-text class="mx-1" >当前记录{{ currentNum }}/{{ table_module.recordTotal }} </el-text>  
              <el-text>
              <el-tag :type="tagType.type">{{ tagType.desc }}</el-tag>
            </el-text>
              <el-text>审核完成：<el-tag :type="auditFinshed?'success':'danger'">{{auditFinshed}}</el-tag> </el-text>
            </el-col>
            <el-col :span="12" style="line-height: 32px;text-align: right;">
              <el-text class="mx-1" style="width: 216px;" >人工审核时间:{{ table_module.currentItem?.manual_audit_time }}
              </el-text>  
              <el-text class="mx-1">审核用户: {{ getUserName(table_module.currentItem?.auditor_id)}}
              </el-text>   
            </el-col>
          </el-row>
        </div>
      </el-header>
      <el-main>
        <!--主内容-->
        <auditview :record="table_module.currentItem" :disabled-audit="true">
          <template #default>
            <div class="navContent">
              <el-button-group class="ml-4">
                <el-button
                  type="primary"
                  @click="onNavClick(-1)"
                  :icon="ArrowLeft"
                  :disabled="!previous_enabled"
                  >上一条
                </el-button>
                <el-button type="primary" @click="onNavClick(1)" :disabled="!next_enabled"
                  >下一条 <el-icon> <ArrowRight /></el-icon>
                </el-button>
                <el-input number style="width: 80px" v-model="currentNum" @keydown.stop>
                  <template #prefix>第</template>
                  <template #suffix> 条 </template>
                </el-input>
              </el-button-group>
           </div>
          </template>
        </auditview>
      </el-main> 
    </el-container>
  </div>
</template>

<script setup lang="ts" name="basetable">
import { ref, reactive, onMounted, computed, watch, onUnmounted } from 'vue'
import { useRoute,useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, ElTag } from 'element-plus'
import { Pointer, ArrowLeft, ArrowRight, Finished } from '@element-plus/icons-vue'
import * as jobs_api from '../api/boat/jobs'
import * as api from '../api/pd'
import * as app_api from '../api/app'
import { showLoading, closeLoading } from '../components/Logining'
import auditview, { AuditType, getAuditTypeDesc, getTagType } from '../components/process/auditview'
import useUserSelect from '../hook/useUserSelect' 

interface Query {
  id: BigInt
}
interface Table_module {
  query: Query
  allData: api.Iorders_record[]
  unAuditIds: number[]
  job?:jobs_api.JobItem
  currentItem?: api.Iorders_record
  currentIndex: number //当前记录
  recordTotal: number //接收到的记录数
}

const table_module = reactive<Table_module>({
  query: {
    id: BigInt(-1)
  },

  allData: [],
  unAuditIds: [],
  currentIndex: 0,
  recordTotal: 0 //获得的记录数
})
const auditFinshed= computed(()=>{ 
  if(table_module.currentItem&&table_module.unAuditIds&&table_module.unAuditIds.length>0){ 
    return !table_module.unAuditIds.includes(table_module.currentItem.id)
  }
  return true
})
const onNavClick = (i: number) => {
  table_module.currentIndex += i
  table_module.currentItem = table_module.allData[table_module.currentIndex]
}
const currentNum = computed({
  get() {
    return table_module.currentIndex + 1
  },
  set(val: number) {
    if (table_module.allData.length == 0) table_module.currentIndex = 0
    else if (val >= table_module.allData.length)
      table_module.currentIndex = table_module.allData.length - 1
    else table_module.currentIndex = val - 1
    table_module.currentItem = table_module.allData[table_module.currentIndex]
  }
})

 
const previous_enabled = computed(() => {
  if (table_module.recordTotal > 0 && table_module.currentIndex != 0) return true
  return false
})
const next_enabled = computed(() => {
  if (table_module.recordTotal > 0 && table_module.currentIndex + 1 != table_module.recordTotal)
    return true
  return false
})
const tagType = ref<{ type: '' | 'info' | 'warning' | 'success' | 'danger'; desc: string }>({
  type: '',
  desc: ''
})

watch(
  () => table_module.currentItem?.manual_audit_result,
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

const {  getUserName } = useUserSelect()

// 获取表格数据
const getData = () => {
  showLoading()
  if (table_module.query.id  )
    jobs_api
      .get_records_svc(table_module.query.id)
      .then((res) => {
        if (res.code == 0) {
          table_module.currentIndex = 0
          table_module.allData = res.data.list
          table_module.unAuditIds=res.data.unAuditIds
          table_module.job=res.data.job
          if (table_module.allData.length > 0) {
            table_module.currentItem = table_module.allData[0]
            table_module.recordTotal = table_module.allData.length || 0
          } else {
            ElMessage.warning('未找到相关数据！')
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

const route = useRoute()
const router=useRouter() 
watch(()=>router.currentRoute.value.params.id,(id)=>{ 
  //push path 时 id 发生改变
  table_module.query.id = BigInt(id as string) 
  getData()
})

onMounted(() => {  
  //第一次加载时运行该代码
  let id=router.currentRoute.value.params.id || route.query.id 
  if (id) {
    table_module.query.id = BigInt(id as string)
    getData()
  } else {
    ElMessage.error("参数不正确！")
  }
})
onUnmounted(()=>{
  console.info("onUnmounted") 
})
 
const app_setting = ref<app_api.ClientConfig>({ batchAudit: false })
app_api.get_config().then((res) => {
  app_setting.value = res.data
})
</script>
<style scoped lang="less">
@import '../assets/css/tables.css';
.el-text{ margin-right: 10px;}
.container-layout {
  &:focus-visible,
  &:focus {
    outline: 0px !important;
  }
}
::v-deep .navContent {
  position:absolute; bottom: 0 ; right: 5px;
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
