import { computed, defineComponent, nextTick, VNodeChild } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import {
  ElButton,
  ElTableColumn,
  ElMessage,
  ElSelect,
  ElOption,
  ElImage,
  ElImageViewer,
  ElMessageBox,
  ElInput,
  ElAffix,
  ElDivider
} from 'element-plus'
import { Search } from '@element-plus/icons-vue'

import { getBaseUrl, PageAllLayouts, hasAuthority } from 'co6co'
import { TableView, TableViewInstance, image2Option } from 'co6co-right'
import style from '@/assets/css/imageView.module.less'
import * as api from '@/api/dev'
export default defineComponent({
  setup(prop, ctx) {
    //:define
    interface IQueryItem {
      date?: string
      name?: string
    }
    const DATA = reactive<{
      title?: string
      query: IQueryItem
      headItemWidth: { width: string }
      selectData: Array<string>
    }>({
      query: {},
      headItemWidth: { width: '180px' },
      selectData: []
    })

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()

    const onSearch = () => {
      viewRef.value?.search()
    }
    const onLoadFolder = () => {
      api.img_folder_select_svc().then((res) => {
        DATA.selectData = res.data
        if (res.data.length > 0) DATA.query.date = DATA.selectData[0]
        onSearch()
      })
    }
    onMounted(async () => {
      onLoadFolder()
    })
    type IData = { options: image2Option[] }
    const dataResult = ref<number[]>([])
    const dataUrlsResult = ref<{ name: string; url: string }[]>([])
    const preview = ref<{ show: boolean; index: number; name?: string }>({ show: false, index: 0 })
    const onShowImage = (name: string, index: number) => {
      preview.value = { show: true, index: index, name: name }
    }
    const urls = computed(() => {
      return dataUrlsResult.value.map((v, i) => {
        return v.url
      })
    })
    const onFilter = (data: Array<string>) => {
      //const result: Array<image2Option> = []
      dataUrlsResult.value = []
      const result: IData[] = []
      for (let i = 0; i < data.length; i++) {
        const item = {
          url: `${getBaseUrl()}/api/dev/img/preview/${DATA.query.date}/${data[i]}`,
          authon: true
        }
        dataUrlsResult.value.push({ url: item.url, name: data[i] })
      }

      return result
    }
    const onDelFolder = (name: string) => {
      ElMessageBox.confirm(`确定删除${name}文件夹吗？`, '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        api.img_del_folder_svc(name).then((res) => {
          ElMessage.success('删除成功')
          onLoadFolder()
        })
      })
    }

    //:page reader
    const rander = (): VNodeChild => {
      return (
        <>
          <style>{` 
          .el-affix{width:100%}
          .el-affix--fixed {
            background: #f0f0f0;
            margin-top:-20px;
            padding-top:8px;
          }
          .el-divider--horizontal{ margin:5px 0 5px 0;}
          .el-footer{height:auto;}
        `}</style>

          <div id={style.imgList} class="affix-container">
            <ElAffix offset={120} target=".affix-container">
              <TableView
                dataApi={api.img_list_svc}
                resultFilter={onFilter}
                ref={viewRef}
                showPaged={true}
                query={DATA.query}
                layouts={PageAllLayouts}
              >
                {{
                  header: () => (
                    <>
                      <div class="handle-box">
                        <ElSelect style={DATA.headItemWidth} v-model={DATA.query.date}>
                          {DATA.selectData.map((v, i) => (
                            <ElOption label={v} key={i} value={v}>
                              <span>{v}</span>
                              <ElButton text onClick={() => onDelFolder(v)}>
                                删除
                              </ElButton>
                            </ElOption>
                          ))}
                        </ElSelect>
                        <ElInput
                          width={160}
                          clearable
                          v-model={DATA.query.name}
                          placeholder="文件名称"
                          class="handle-input"
                        />

                        <ElButton type="primary" icon={Search} onClick={onSearch}>
                          搜索
                        </ElButton>
                      </div>
                    </>
                  ),
                  default: () => (
                    <>
                      <ElTableColumn />
                    </>
                  ),
                  footer: () => (
                    <>
                      <ElDivider />
                    </>
                  )
                }}
              </TableView>
            </ElAffix>
            {dataUrlsResult.value.map((v, i) => {
              return (
                <div class="block">
                  <span class="demonstration">{v.name}</span>
                  <ElImage
                    key={i}
                    src={v.url}
                    lazy
                    zoomRate={1.2}
                    maxScale={7}
                    minScale={0.2}
                    showPropress
                    onClick={() => onShowImage(v.name, i)}
                  />
                </div>
              )
            })}
            {preview.value && preview.value.show ? (
              <ElImageViewer
                urlList={urls.value}
                show-progress
                initialIndex={preview.value.index}
                onClose={() => (preview.value.show = false)}
              />
            ) : (
              <div />
            )}
          </div>
        </>
      )
    }
    return rander
  } //end setup
})
