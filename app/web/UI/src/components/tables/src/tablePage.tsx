import { defineComponent, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElTable, ElContainer, ElMain, ElHeader, ElScrollbar, ElFooter } from 'element-plus'
import {
  showLoading,
  closeLoading,
  type IPageParam,
  onColChange,
  type Table_Module_Base,
  Pagination,
  IPageResponse
} from 'co6co'
export default defineComponent({
  props: {
    dataApi: {
      type: Promise<IPageResponse>,
      required: true
    }
  },
  setup(prop, ctx) {
    //:define
    interface IQueryItem extends IPageParam {
      appid?: string
      title?: string
    }
    interface Table_Module extends Table_Module_Base {
      query: IQueryItem
      data: any[]
      currentItem?: any
    }
    //:use
    //const { getPermissKey } = routeHook.usePermission()
    //const dictHook = useDictHook.useDictSelect()
    //const store = get_store()
    //end use

    //:page
    const tableRef = ref<InstanceType<typeof ElTable>>()
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
      prop.dataApi
        .then((res) => {
          DATA.data = res.data
          DATA.pageTotal = res.total || -1
        })
        .finally(() => {
          closeLoading()
        })
    }
    const refesh = () => {
      DATA.query.pageIndex = 1
      queryData()
    }
    const onColChange2 = (column: any) => {
      onColChange(column, DATA.query, queryData)
    }
    //end page
    onMounted(async () => {
      queryData()
    })
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <div class="container-layout">
          <ElContainer>
            <ElHeader>{ctx.slots.header?.()}</ElHeader>
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
                  {ctx.slots.default?.()}
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
          {ctx.slots.footer?.()}
        </div>
      )
    }

    ctx.expose({
      refesh
    })
    rander.refesh = refesh
    return rander
  } //end setup
})