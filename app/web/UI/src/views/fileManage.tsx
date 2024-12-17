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
  ElTooltip,
  ElIcon
} from 'element-plus'
import {
  Search,
  Plus,
  Edit,
  ArrowLeftBold,
  Refresh,
  Download as DownloadIcon,
  Loading
} from '@element-plus/icons-vue'
import style from '@/assets/css/file.module.less'
import { FormOperation, byte2Unit } from 'co6co'
import axios from 'axios'
import {
  routeHook,
  DictSelectInstance,
  tableScope,
  TableView,
  TableViewInstance,
  Download,
  download_header_svc,
  getFileName
} from 'co6co-right'

import Diaglog from '../components/biz/modifyTask'

import { list_svc, getResourceUrl, list_param, list_res as Item } from '@/api/file'

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
    const onClickSubFolder = (path: string) => {
      DATA.query.root = path
      onSearch()
    }
    const onClickClcFolder = (row: Item & { loading?: boolean }) => {
      row.loading = true
      download_header_svc(getResourceUrl(row.path, true), true)
        .then((res) => {
          row.size = Number(res.headers['content-length'])
        })
        .finally(() => {
          row.loading = false
        })
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
                      onChange={onSearch}
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
                          <ElLink onClick={() => onClickSubFolder(scope.row.path)}>
                            {scope.row.name}
                          </ElLink>
                        )}
                      </ElTooltip>
                    )
                  }}
                </ElTableColumn>
                <ElTableColumn label="大小" prop="name" align="center" width={180}>
                  {{
                    default: (scope: { row: Item & { loading?: boolean } }) =>
                      scope.row.isFile ? (
                        byte2Unit(scope.row.size, 'b', 2)
                      ) : (
                        <ElLink onClick={() => onClickClcFolder(scope.row)}>
                          <ElButton text loading={scope.row.loading}>
                            {scope.row.size ? byte2Unit(scope.row.size, 'b', 2) : '计算'}
                          </ElButton>
                        </ElLink>
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
                        <Download
                          authon
                          showPercentage
                          chunkSize={2 * 1024 * 1024}
                          url={getResourceUrl(scope.row.path, scope.row.isFile)}
                          v-permiss={getPermissKey(routeHook.ViewFeature.download)}
                        />
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
