import { defineComponent, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElTag, ElButton, ElInput, ElTable, ElTableColumn, ElSelect, ElOption } from 'element-plus'
import { Search, Sugar, View } from '@element-plus/icons-vue'

import { type IPageParam, type Table_Module_Base, getTableIndex } from 'co6co'
import { routeHook, useDictHook } from 'co6co-right'

import { DictTypeCodes } from '../api/app'
import Diaglog, { type Item } from '../components/biz/modifyTask'
import { tableView } from '../components/tables'
import { task as api } from '../api/biz'

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
    const useCategory = useDictHook.useDictSelect()
    const useStatue = useDictHook.useDictSelect()
    const useState = useDictHook.useDictSelect()
    //end use

    //:page
    const viewRef = ref<InstanceType<typeof tableView>>()
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

    const onOpenDialog = (row: Item) => {
      DATA.diaglogTitle = `查询[${row.title}]详情`
      DATA.currentItem = row
      diaglogRef.value?.openDialog(row.id)
    }

    onMounted(async () => {
      await store.refesh()
      viewRef.value?.refesh()
    })

    //:page reader
    const rander = (): VNodeChild => {
      return (
        <tableView dataApi={api.get_table_svc}>
          {{
            header: () => (
              <>
                <div class="handle-box">
                  <ElInput
                    style="width: 160px"
                    clearable
                    v-model={DATA.query.title}
                    placeholder="模板标题"
                    class="handle-input"
                  />
                  <ElSelect
                    style="width: 160px"
                    v-model={DATA.query.appid}
                    placeholder="所属公众号"
                  >
                    {store.list.map((item, index) => {
                      return (
                        <>
                          <ElOption key={index} label={item.name} value={item.openId}></ElOption>
                        </>
                      )
                    })}
                  </ElSelect>
                  <ElButton
                    type="primary"
                    icon={Search}
                    onClick={() => {
                      viewRef.value?.refesh()
                    }}
                  >
                    搜索
                  </ElButton>
                </div>
              </>
            ),
            default: () => (
              <>
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
              </>
            ),
            footer: () => (
              <>
                <Diaglog ref={diaglogRef} title={DATA.diaglogTitle}></Diaglog>
              </>
            )
          }}
        </tableView>
      )
    }
    return rander
  } //end setup
})
