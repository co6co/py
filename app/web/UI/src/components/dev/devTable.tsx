import { defineComponent, VNodeChild, PropType, computed } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElTag, ElButton, ElInput, ElTableColumn, ElButtonGroup, ElMessageBox, ElLink, ElIcon, ElRow } from 'element-plus'
import { Search, Plus, Edit, Delete, View, ArrowDown, MoreFilled } from '@element-plus/icons-vue'

import { FormOperation, IResponse, EnumSelect, IEnumSelect, PageLayouts } from 'co6co'
import {
  routeHook,
  tableScope,
  TableView,
  deleteHook,
  ViewFeature,
  TableViewInstance,
  StateSelect,
  StateSelectInstance,
  Download,
  UploadFile
} from 'co6co-right'

import Diaglog, { type Item } from '@/components/dev/modifyDev'
import ImgViewDialog from './imgViewDialog'

import * as api from '@/api/dev'

export const features = {
  add: ViewFeature.add,
  edit: ViewFeature.edit,
  del: ViewFeature.del,
  downCheckResult: { text: "下载检测结果", value: "downCheckResult" }
}
export default defineComponent({
  props: {
    dataApi: {
      type: Function as PropType<(data: any) => Promise<IResponse<any>>>
    },
    allowImport: {
      type: Boolean,
      default: false
    },
    hasOpertion: {
      type: Boolean,
      default: false
    },
    autoLoadData: {
      type: Boolean,
      default: true
    },
    extra: {
      type: Object as PropType<{ [key: string]: any }>,
      default: {}
    }
  },
  setup(prop, ctx) {
    //:define
    interface IQueryItem {
      [key: string]: any
      state?: number
      name?: string
      code?: string
      category?: number
      ip?: string
      checkDesc?: string
    }
    const DATA = reactive<{
      title?: string
      query: IQueryItem
      currentItem?: Item
      headItemWidth: { width: string }
      queryMoreOption: boolean,
    }>({
      query: {},
      headItemWidth: { width: '180px' },
      queryMoreOption: false
    })
    //:use
    const { getPermissKey } = routeHook.usePermission()

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()
    const diaglogRef = ref<InstanceType<typeof Diaglog>>()
    const statueInstanceRef = ref<StateSelectInstance>()

    const imgViewDialogRef = ref<InstanceType<typeof ImgViewDialog>>()
    const onOpenDialog = (row?: Item) => {
      DATA.title = row ? `编辑[${row?.name}]` : '增加'
      DATA.currentItem = row
      diaglogRef.value?.openDialog(row ? FormOperation.edit : FormOperation.add, row)
    }
    const imgPath = ref("")
    const onImageOpenDialog = (row?: Item) => {
      if (row?.checkImgPath) {
        imgPath.value = row?.checkImgPath
        imgViewDialogRef.value?.openDialog(row.name)
      }
      else {
        imgPath.value = ""
        ElMessageBox.alert("未找到设备巡检的图片路径")
      }
    }
    const onSearch = () => {
      viewRef.value?.search()
    }
    const onRefesh = () => {
      viewRef.value?.refesh()
    }
    const queryparam = computed(() => {
      return { ...DATA.query, ...prop.extra }
    })

    const DeviceCategory = ref<IEnumSelect[]>([])
    const getCategoryName = (v: number) => {
      return DeviceCategory.value?.find((item) => item.value === v)?.label
    }
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
        <TableView
          dataApi={prop.dataApi}
          ref={viewRef}
          query={queryparam.value}
          layouts={PageLayouts}
          autoLoadData={prop.autoLoadData}
        >
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
                    v-model={DATA.query.ip}
                    placeholder="设备IP地址"
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

                  <ElLink type="primary" onClick={() => { DATA.queryMoreOption = !DATA.queryMoreOption }}>
                    <ElIcon size={20}><MoreFilled /></ElIcon>
                  </ElLink>

                  <ElButton type="primary" icon={Search} onClick={onSearch}>
                    搜索
                  </ElButton>
                  {prop.hasOpertion ? (
                    <ElButton
                      type="primary"
                      icon={Plus}
                      v-permiss={getPermissKey(ViewFeature.add)}
                      onClick={() => {
                        onOpenDialog()
                      }}
                    >
                      新增
                    </ElButton>
                  ) : (
                    <></>
                  )}

                  <Download v-permiss={features.downCheckResult.value} authon title="下载检测结果" url={api.getCheckDataUrl()} text={false} />
                </div>
                {DATA.queryMoreOption ? (<ElRow>
                  <div class="handle-box">
                    <div class="formItem">
                      <ElInput
                        style={DATA.headItemWidth}
                        clearable
                        v-model={DATA.query.code}
                        placeholder="设备代码"
                        class="handle-input"
                      />
                      <ElInput
                        style={DATA.headItemWidth}
                        clearable
                        v-model={DATA.query.checkDesc}
                        placeholder="检测状态"
                        class="handle-input"
                      />
                      {prop.allowImport ? (
                        <ElButtonGroup style="margin-left:10px">
                          <Download authon title="下载模板" url={api.getResourceUrl()} text={false} />
                          <UploadFile
                            text="上传数据"
                            accept=".xlsx,.xls"
                            onSuccess={onSuccess}
                            uploadApi={api.upload_template}
                          />
                        </ElButtonGroup>
                      ) : (
                        <>
                        </>
                      )}
                    </div>
                  </div>
                </ElRow>) : <></>}


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
                  label="名称"
                  prop="name"
                  align="center"
                  sortable="custom"
                  width={180}
                  showOverflowTooltip={true}
                />
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
                  label="设备类型"
                  prop="category"
                  align="center"
                  width={180}
                  sortable="custom"
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTag>{getCategoryName(scope.row.category)}</ElTag>
                    )
                  }}
                </ElTableColumn>
                <ElTableColumn
                  label="网络地址"
                  prop="ip"
                  width="160"
                  align="center"
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="检测结果"
                  prop="checkDesc"
                  align="left"
                  sortable="custom"
                  width={120}
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="检测时间"
                  prop="checkTime"
                  align="center"
                  sortable="custom"
                  width={160}
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="设备厂家"
                  prop="vender"
                  align="center"
                  sortable="custom"
                  width={120}
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="代码"
                  prop="code"
                  align="center"
                  width={180}
                  sortable="custom"
                  showOverflowTooltip={true}
                />
                <ElTableColumn
                  label="序列号"
                  prop="serialNumber"
                  align="center"
                  sortable="custom"
                  width={120}
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
                  prop="createTime"
                  label="创建时间"
                  sortable="custom"
                  width={160}
                  show-overflow-tooltip={true}
                />
                <ElTableColumn
                  prop="updateTime"
                  label="更新时间"
                  sortable="custom"
                  width={160}
                  show-overflow-tooltip={true}
                />
                {prop.hasOpertion ? (
                  <ElTableColumn label="操作" width={320} align="left" fixed="right">
                    {{
                      default: (scope: tableScope<Item>) => (
                        <>
                          <ElButton
                            text={true}
                            icon={Edit}
                            onClick={() => onOpenDialog(scope.row)}
                            v-permiss={getPermissKey(ViewFeature.edit)}
                          >
                            编辑
                          </ElButton>
                          <ElButton
                            text={true}
                            icon={Delete}
                            onClick={() => onDelete(scope.row)}
                            v-permiss={getPermissKey(ViewFeature.del)}
                          >
                            删除
                          </ElButton>
                          {
                             scope.row.category < 100 ? (
                              <ElButton
                                text={true}
                                icon={View}
                                onClick={() => onImageOpenDialog(scope.row)}
                              >
                                查看图片
                              </ElButton>
                            ) : (
                              <></>
                            )

                          }

                        </>
                      )
                    }}
                  </ElTableColumn>
                ) : (
                  <></>
                )}
              </>
            ),
            footer: () => (
              <>
                <Diaglog ref={diaglogRef} title={DATA.title} onSaved={onRefesh} />
                <ImgViewDialog ref={imgViewDialogRef} path={imgPath.value} />
              </>
            )
          }}
        </TableView>
      )
    }
    ctx.expose({
      refesh: onRefesh
    })
    rander.refesh = onRefesh
    return rander
  } //end setup
})
