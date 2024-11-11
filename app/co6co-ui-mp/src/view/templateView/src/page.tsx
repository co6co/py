import { defineComponent, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import {
  ElTag,
  ElButton,
  ElInput,
  ElTable,
  ElTableColumn,
  ElContainer,
  ElMain,
  ElHeader,
  ElScrollbar, 
  ElFooter,
  ElSelect,
  ElOption,
  ElMessage,
  ElMessageBox
} from 'element-plus'
import { Search, Sugar, View } from '@element-plus/icons-vue'

import {
  showLoading,
  closeLoading,
  type IPageParam,
  onColChange,
  type Table_Module_Base,
  getTableIndex,
  Pagination
} from 'co6co'
import { routeHook } from 'co6co-right'
 
import { get_store } from 'co6co-wx'
import Diaglog, { type TemplateItem as  Item } from '@/components/templateInfo'

import {template as api} from '@/api'

export default defineComponent({
  setup(prop, ctx) {
    //:define
    interface IQueryItem extends IPageParam {
      appid?: string
      title?: string
    }
    interface Table_Module extends Table_Module_Base {
      query: IQueryItem
      data: Item[]
      currentItem?: Item
    }
    //:use
    const { getPermissKey } = routeHook.usePermission()
    //const dictHook = useDictHook.useDictSelect()
    const store = get_store()
    //end use

    //:page
    const tableRef = ref<InstanceType<typeof ElTable>>()
    const diaglogRef = ref<InstanceType<typeof Diaglog>>()
    const DATA = reactive<Table_Module>({
      query: {
        pageIndex: 1,
        pageSize: 10,
        order: 'asc',
        orderBy: ''
      },
      data: [],
      pageTotal: -1,
      diaglogTitle: ''
    })
    // 获取表格数据
    const queryData = () => {
      showLoading()
      api
        .get_table_svc(DATA.query)
        .then((res) => {
          DATA.data = res.data
          DATA.pageTotal = res.total || -1
        })
        .finally(() => {
          closeLoading()
        })
    }
    const onSearch = () => {
      DATA.query.pageIndex = 1
      queryData()
    }
    const onColChange2 = (column: any) => {
      onColChange(column, DATA.query, queryData)
    }

    const onSyncTemplage = () => {
      if (!DATA.query.appid) {
        ElMessageBox({
          title: '提示',
          type: 'warning',
          message: '请选择同步得公众号!'
        })
        return
      }
      showLoading()
      api
        .sync_svc(DATA.query.appid)
        .then((res) => {
          ElMessage.success(res.message)
          queryData()
        })
        .finally(() => {
          closeLoading()
        })
    }

    const onOpenDialog = (row: Item) => {
      DATA.diaglogTitle = `查询[${row.title}]详情`
      DATA.currentItem = row
      diaglogRef.value?.openDialog(row.id)
    }

    //end page

    onMounted(async () => {
      await store.refesh() 
      queryData()
    })

    //:page reader
    const rander = (): VNodeChild => {
      return (
        <div class="container-layout">
          <ElContainer>
            <ElHeader>
              <div class="handle-box">
                <ElInput
                  style="width: 160px"
                  clearable
                  v-model={DATA.query.title}
                  placeholder="模板标题"
                  class="handle-input"
                />
                <ElSelect style="width: 160px" clearable v-model={DATA.query.appid} placeholder="所属公众号">
                  {store.list.map((item, index) => {
                    return (
                      <>
                        <ElOption key={index} label={item.name} value={item.openId}></ElOption>
                      </>
                    )
                  })}
                </ElSelect>
                <ElButton type="primary" icon={Search} onClick={onSearch}>
                  搜索
                </ElButton>
                <ElButton
                  icon={Sugar}
                  type="info"
                  v-permiss={getPermissKey(routeHook.ViewFeature.get)}
                  onClick={onSyncTemplage}
                >
                  同步
                </ElButton>
              </div>
            </ElHeader>
            <ElMain>
              <ElScrollbar>
                <ElTable
                  data={DATA.data}
                  border={true}
                  class="table"
                  ref={tableRef}
                  headerCellClassName="table-header"
                  onSort-change={onColChange2}
                >
                  <ElTableColumn label="序号" width={55} align="center">
                    {{ default: (scope: any) => <>{scope.$index + 1}</> }}
                  </ElTableColumn>
                  <ElTableColumn label="序号" width={55} align="center">
                    {{
                      default: (scope: any) => <>{getTableIndex(DATA.query, scope.$index)}</>
                    }}
                  </ElTableColumn>
                  <ElTableColumn
                    label="编号"
                    prop="templateId"
                    align="center"
                    width={180}
                    sortable="custom"
                    showOverflowTooltip={true}
                  />
                  <ElTableColumn
                    label="标题"
                    prop="title"
                    align="center"
                    sortable="custom"
                    showOverflowTooltip={true}
                  />
                  <ElTableColumn
                    label="所属公众号"
                    prop="openId"
                    sortable="custom"
                    align="center"
                    showOverflowTooltip={true}
                  >
                    {{
                      default: (scope: any) => (
                        <ElTag>{store.getItem(scope.row.ownedAppid)?.name}</ElTag>
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
                  <ElTableColumn label="操作" width={200} align="center" fixed="right">
                    {{
                      default: (scope: any) => (
                        <ElButton
                          text={true}
                          icon={View}
                          onClick={() => onOpenDialog(scope.row)}
                          v-permiss={getPermissKey(routeHook.ViewFeature.view)}
                        >
                          查看
                        </ElButton>
                      )
                    }}
                  </ElTableColumn>
                </ElTable>
              </ElScrollbar>
            </ElMain>
            <ElFooter>
              <Pagination
                option={DATA.query}
                total={DATA.pageTotal}
                onCurrentPageChange={queryData}
                onSizeChage={queryData}
              ></Pagination>
            </ElFooter>
          </ElContainer>
          <Diaglog ref={diaglogRef} title={DATA.diaglogTitle}></Diaglog>
        </div>
      )
    }
    return rander
  } //end setup
})
