import { computed, defineComponent, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import {
  ElTag,
  ElButton,
  ElInput,
  ElTableColumn,
  ElMessage,
  ElRow,
  ElCol,
  ElLink,
  ElTooltip
} from 'element-plus'
import { Search, Plus, Edit, ArrowLeftBold, Refresh, Download } from '@element-plus/icons-vue'
import style from '@/assets/css/file.module.less'
import { FormOperation, byte2Unit } from 'co6co'

import {
  routeHook,
  DictSelectInstance,
  tableScope,
  TableView,
  TableViewInstance
} from 'co6co-right'

import Diaglog from '../components/biz/modifyTask'

import { list_svc, list_param, list_res as Item } from '@/api/file'

export default defineComponent({
  setup(prop, ctx) {
    const DATA = reactive<{ query: list_param; currentItem?: Item; split: RegExp }>({
      query: { root: 'I:' },
      split: /[/\\]/
    })

    //:use
    const { getPermissKey } = routeHook.usePermission()

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()
    const diaglogRef = ref<InstanceType<typeof Diaglog>>()
    const stateDictRef = ref<DictSelectInstance>()
    const statusDictRef = ref<DictSelectInstance>()

    const onOpenDialog = (row?: Item) => {
      DATA.currentItem = row
      diaglogRef.value?.openDialog(row ? FormOperation.edit : FormOperation.add, row)
    }
    const onSearch = () => {
      viewRef.value?.search()
    }
    const onRefesh = () => {
      viewRef.value?.refesh()
    }
    onMounted(async () => {
      onSearch()
    })
    const isEditing = ref(false)
    const handleFocus = () => {
      isEditing.value = true
    }
    const handleBlur = () => {
      isEditing.value = false
    }
    const previewRoot = computed(() => {
      if (DATA.query.root) {
        const arr = DATA.query.root.split(DATA.split)
        return '根目录' + arr.join(' > ').substring(1)
      }
      return ''
    })
    const onRootUp = () => {
      if (DATA.query.root) {
        const arr = DATA.query.root.split(DATA.split)
        const result = arr.slice(0, arr.length - 1)
        console.info(arr, result)
        if (result.length == 1 && result[0] == '') DATA.query.root = '/'
        else DATA.query.root = result.join('/')
        onSearch()
      }
    }
    const ontresultFileter = (data: { res: any[]; root: string }) => {
      DATA.query.root = data.root
      return data.res
    }
    const onClickSubFolder = (sub: string) => {
      if (DATA.query.root == '/') DATA.query.root = '/' + sub
      else DATA.query.root = DATA.query.root + '/' + sub
      onSearch()
    }
    const onDownload = (filePath: string) => {
      alert(filePath)
    }
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <TableView
          dataApi={list_svc}
          ref={viewRef}
          query={DATA.query}
          showPaged={false}
          resultFilter={ontresultFileter}
        >
          {{
            header: () => (
              <>
                <ElRow>
                  <ElCol span={12}>
                    <ElInput
                      v-model={DATA.query.root}
                      class={isEditing.value ? style.editor : style.show}
                      value={isEditing.value ? DATA.query.root : previewRoot.value}
                      onFocus={handleFocus}
                      onBlur={handleBlur}
                    >
                      {{
                        prepend: () => (
                          <ElButton title="上级目录" icon={ArrowLeftBold} onClick={onRootUp} />
                        ),
                        append: () => <ElButton title="刷新" icon={Refresh} onClick={onSearch} />
                      }}
                    </ElInput>
                  </ElCol>
                  <ElCol span={6} offset={6}>
                    <ElInput
                      style="width: 160px"
                      clearable
                      v-model={DATA.query.name}
                      placeholder="搜索文件/目录"
                      class="handle-input"
                    />
                    <ElButton type="primary" icon={Search} onClick={onSearch}>
                      搜索
                    </ElButton>
                  </ElCol>
                </ElRow>
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
                  width={180}
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTooltip effect="dark" content={scope.row.path} showAfter={1500}>
                        {scope.row.isFile ? (
                          scope.row.name
                        ) : (
                          <ElLink onClick={() => onClickSubFolder(scope.row.name)}>
                            {scope.row.name}
                          </ElLink>
                        )}
                      </ElTooltip>
                    )
                  }}
                </ElTableColumn>
                <ElTableColumn label="大小" prop="name" align="center" width={180}>
                  {{
                    default: (scope: { row: Item }) =>
                      scope.row.isFile ? (
                        byte2Unit(scope.row.size, 'b', 2)
                      ) : (
                        <ElLink onClick={() => onClickSubFolder(scope.row.name)}>计算</ElLink>
                      )
                  }}
                </ElTableColumn>

                <ElTableColumn
                  label="cron[s]"
                  width="160"
                  prop="cron"
                  align="center"
                  showOverflowTooltip={true}
                />
                <ElTableColumn label="状态" prop="state" align="center" showOverflowTooltip={true}>
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTag>{stateDictRef.value?.getName(scope.row.state)}</ElTag>
                    )
                  }}
                </ElTableColumn>
                <ElTableColumn
                  label="运行状态"
                  prop="execStatus"
                  align="center"
                  showOverflowTooltip={true}
                >
                  {{
                    default: (scope: { row: Item }) => (
                      <ElTag>{statusDictRef.value?.getName(scope.row.execStatus)}</ElTag>
                    )
                  }}
                </ElTableColumn>

                <ElTableColumn
                  prop="updateTime"
                  label="修改时间"
                  width={160}
                  show-overflow-tooltip={true}
                />
                <ElTableColumn label="操作" width={260} align="center" fixed="right">
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
                        {scope.row.isFile ? (
                          <ElButton
                            text={true}
                            icon={Download}
                            onClick={() => onDownload(scope.row.path)}
                            v-permiss={getPermissKey(routeHook.ViewFeature.download)}
                          >
                            下载
                          </ElButton>
                        ) : (
                          <></>
                        )}
                      </>
                    )
                  }}
                </ElTableColumn>
              </>
            ),
            footer: () => (
              <>
                <Diaglog ref={diaglogRef} title={DATA.query.root} onSaved={onRefesh} />
              </>
            )
          }}
        </TableView>
      )
    }
    return rander
  } //end setup
})
