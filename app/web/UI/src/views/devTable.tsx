import { defineComponent, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElTag, ElButton, ElInput, ElTableColumn, ElButtonGroup, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete, UploadFilled } from '@element-plus/icons-vue'

import { FormOperation, IResponse, EnumSelect, IEnumSelect } from 'co6co'
import {
  routeHook,
  DictSelectInstance,
  tableScope,
  TableView,
  deleteHook,
  TableViewInstance,
  StateSelect,
  StateSelectInstance,
  useDictHook,
  Download,
  UploadFile
} from 'co6co-right'

import Diaglog, { type Item } from '@/components/dev/modifyDev'

import * as api from '@/api/dev'
export default defineComponent({
  setup(prop, ctx) {
    //:define
    interface IQueryItem {
      state?: number
      name?: string
      code?: string
      category?: number
    }
    const DATA = reactive<{
      title?: string
      query: IQueryItem
      currentItem?: Item
      headItemWidth: { width: string }
    }>({
      query: {},
      headItemWidth: { width: '180px' }
    })

    //:use
    const { getPermissKey } = routeHook.usePermission()

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()
    const diaglogRef = ref<InstanceType<typeof Diaglog>>()
    const statueInstanceRef = ref<StateSelectInstance>()
    const onOpenDialog = (row?: Item) => {
      DATA.title = row ? `编辑[${row?.name}]` : '增加'
      DATA.currentItem = row
      diaglogRef.value?.openDialog(row ? FormOperation.edit : FormOperation.add, row)
    }
    const onSearch = () => {
      viewRef.value?.search()
    }
    const onRefesh = () => {
      viewRef.value?.refesh()
    }
    const DeviceCategory = ref<IEnumSelect[]>([])
    onMounted(async () => {
      const res = await api.dev_category_svc()
      DeviceCategory.value = res.data
      onSearch()
    })
    const { deleteSvc } = deleteHook.default(api.del_svc, () => {
      onRefesh()
    })
    const onDelete = (row: Item) => {
      deleteSvc(row.id, row.name)
    }

    const getName = (v: number) => {
      return statueInstanceRef.value?.getName(v)
    }
    const getTagType = (v: number) => {
      return statueInstanceRef.value?.getTagType(v)
    }
    const onSuccess = (data: IResponse) => {
      ElMessageBox.alert(data.message, '提示', {
        type: 'success',
        confirmButtonText: '确定'
      })
      onSearch()
    }

    //:page reader
    const rander = (): VNodeChild => {
      return (
        <TableView dataApi={api.get_table_svc} ref={viewRef} query={DATA.query}>
          {{
            header: () => (
              <>
                <div class="handle-box">
                  <ElInput
                    style={DATA.headItemWidth}
                    clearable
                    v-model={DATA.query.name}
                    placeholder="设备名称"
                    class="handle-input"
                  />
                  <ElInput
                    style={DATA.headItemWidth}
                    clearable
                    v-model={DATA.query.code}
                    placeholder="设备代码"
                    class="handle-input"
                  />
                  <EnumSelect
                    style={DATA.headItemWidth}
                    data={DeviceCategory.value}
                    v-model={DATA.query.category}
                    placeholder="设备类型"
                  />
                  <StateSelect
                    ref={statueInstanceRef}
                    style={DATA.headItemWidth}
                    v-model={DATA.query.state}
                    placeholder="状态"
                  />
                  <ElButton type="primary" icon={Search} onClick={onSearch}>
                    搜索
                  </ElButton>
                  <ElButton
                    type="primary"
                    icon={Plus}
                    v-permiss={getPermissKey(routeHook.ViewFeature.add)}
                    onClick={() => {
                      onOpenDialog()
                    }}
                  >
                    新增
                  </ElButton>
                  <ElButtonGroup style="margin-left:10px">
                    <Download authon title="下载模板" url={api.getResourceUrl()} text={false} />
                    <UploadFile
                      accept=".xlsx,.xls"
                      onSuccess={onSuccess}
                      uploadApi={api.upload_template}
                    />
                  </ElButtonGroup>
                </div>
              </>
            ),
            default: () => (
              <>
                <ElTableColumn label="序号" width={55} align="center">
                  {{
                    default: (scope: tableScope) => viewRef.value?.rowIndex(scope.$index)
                  }}
                </ElTableColumn>
                <ElTableColumn
                  label="编号"
                  prop="code"
                  align="center"
                  width={180}
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="名称"
                  prop="name"
                  align="center"
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="网络地址"
                  prop="ip"
                  width="160"
                  align="center"
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="经纬度"
                  width="160"
                  prop="lat"
                  align="center"
                  sortable="custom"
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTag>{`${scope.row.lng},${scope.row.lat}`}</ElTag>
                    )
                  }}
                </ElTableColumn>
                <ElTableColumn
                  label="状态"
                  prop="state"
                  sortable="custom"
                  align="center"
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTag type={getTagType(scope.row.state)}>{getName(scope.row.state)}</ElTag>
                    )
                  }}
                </ElTableColumn>

                <ElTableColumn
                  prop="createTime"
                  label="创建时间"
                  sortable="custom"
                  width={160}
                  show-overflow-tooltip={true}
                ></ElTableColumn>
                <ElTableColumn
                  prop="updateTime"
                  label="更新时间"
                  sortable="custom"
                  width={160}
                  show-overflow-tooltip={true}
                ></ElTableColumn>
                <ElTableColumn label="操作" width={320} align="center" fixed="right">
                  {{
                    default: (scope: tableScope<Item>) => (
                      <>
                        <ElButton
                          text={true}
                          icon={Edit}
                          onClick={() => onOpenDialog(scope.row)}
                          v-permiss={getPermissKey(routeHook.ViewFeature.edit)}
                        >
                          编辑
                        </ElButton>
                        <ElButton
                          text={true}
                          icon={Delete}
                          onClick={() => onDelete(scope.row)}
                          v-permiss={getPermissKey(routeHook.ViewFeature.del)}
                        >
                          删除
                        </ElButton>
                      </>
                    )
                  }}
                </ElTableColumn>
              </>
            ),
            footer: () => (
              <>
                <Diaglog ref={diaglogRef} title={DATA.title} onSaved={onRefesh} />
              </>
            )
          }}
        </TableView>
      )
    }
    return rander
  } //end setup
})
