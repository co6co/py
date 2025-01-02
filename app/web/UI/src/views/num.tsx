import { computed, defineComponent, nextTick, VNodeChild, watch } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import {
  ElButton,
  ElContainer,
  ElMain,
  ElForm,
  ElScrollbar,
  ElFormItem,
  ElInputNumber,
  ElCheckboxGroup,
  ElCheckboxButton,
  ElRow,
  ElCol
} from 'element-plus'
import { TableViewInstance, DictSelect } from 'co6co-right'
import { IEnumSelect, minus } from 'co6co'
import { get_category_desc_svc, clc_svc, category_desc } from '@/api/tool/num'
import style from '@/assets/css/num.module.less'
export default defineComponent({
  setup(prop, ctx) {
    const DATA = reactive<{
      _selectCount: number
      _cateselectData: IEnumSelect[]
      _category_desc?: category_desc
      category: number
      _selectList: Array<number>
      //选中的值
      list: Array<number>
      dans?: Array<number>

      result: Array<String>
      count: number
    }>({
      _selectCount: 30,
      _cateselectData: [],
      _selectList: [1, 2, 3, 4],
      category: 0,
      list: [],
      result: [],
      count: 0
    })
    //:use

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()

    const onSearch = () => {
      viewRef.value?.search()
    }

    const onSelectCountChanged = (n) => {
      DATA._selectList = []
      for (let i = 1; i <= n; i++) DATA._selectList.push(i)
    }

    const onAllowSelect = (val: number) => {
      if (DATA.dans?.includes(val)) return false
      if (DATA.list?.includes(val)) return true
      return DATA._category_desc && DATA.list.length < DATA._category_desc.select
    }

    const onAllowDanSelect = (val: number) => {
      if (DATA.dans?.includes(val)) return true
      else
        return (
          (!DATA.dans && DATA._category_desc && DATA._category_desc.dan > 0) ||
          (DATA._category_desc && DATA.dans && DATA.dans.length < DATA._category_desc.dan)
        )
    }
    const allowSubmit = computed(() => {
      if (!DATA.dans) DATA.dans = []
      return (
        DATA.list.length == DATA._category_desc?.select &&
        DATA.dans?.length == DATA._category_desc.dan
      )
    })
    const onCalc = () => {
      clc_svc(DATA.category, { list: DATA.list, dans: DATA.dans }).then((res) => {
        DATA.result = res.data.list.map((m) => m.join() + '\r\n')
        DATA.count = res.data.count
      })
    }
    const onClear = () => {
      DATA.list = []
      DATA.dans = []
      DATA.result = []
      DATA.count = 0
    }
    const onCategoryChange = async (n: number) => {
      try {
        const res = await get_category_desc_svc(n)
        DATA._category_desc = res.data
      } catch (e) {
        console.error(e)
      }
    }
    const onDanChanged = (dans: Array<number>) => {
      if (dans.length > 0) DATA.list = minus(DATA.list, dans) as number[]
    }
    onMounted(async () => {
      //const res = await get_category_svc()
      //DATA._cateselectData = res.data
      //await onCategoryChange(DATA._cateselectData[0].value)
      onSelectCountChanged(DATA._selectCount)
      onSearch()
    })
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <ElContainer id={style.view}>
          <ElMain>
            <ElForm labelWidth={100}>
              <ElRow>
                <ElCol span={12}>
                  <ElFormItem label="数量">
                    <ElInputNumber
                      v-model={DATA._selectCount}
                      placeholder="数量"
                      onChange={onSelectCountChanged}
                    />
                  </ElFormItem>
                </ElCol>
                <ElCol span={12}>
                  <ElFormItem label="类型">
                    <DictSelect
                      dictTypeCode="USER_NUM"
                      onChange={onCategoryChange}
                      v-model={DATA.category}
                      valueUseFiled="id"
                      placeholder="类别"
                    />
                  </ElFormItem>
                </ElCol>
              </ElRow>
              {DATA._category_desc && DATA._category_desc.dan > 0 ? (
                <ElFormItem label="胆值">
                  <ElCheckboxGroup v-model={DATA.dans} onChange={onDanChanged}>
                    {DATA._selectList.map((val) => (
                      <>
                        <ElCheckboxButton
                          disabled={!onAllowDanSelect(val)}
                          value={val}
                          class="is-circle buttons-success"
                        >
                          {val}
                        </ElCheckboxButton>
                      </>
                    ))}
                  </ElCheckboxGroup>
                </ElFormItem>
              ) : (
                <></>
              )}
              <ElFormItem label="旋转值">
                <ElCheckboxGroup v-model={DATA.list}>
                  {DATA._selectList.map((val) => (
                    <>
                      <ElCheckboxButton
                        value={val}
                        disabled={!onAllowSelect(val)}
                        class="is-circle buttons-success"
                      >
                        {val}
                      </ElCheckboxButton>
                    </>
                  ))}
                </ElCheckboxGroup>
              </ElFormItem>
              <ElFormItem label={`结果[${DATA.count}]`}>
                <ElScrollbar>
                  <pre style="height:200px">{DATA.result.map((r) => r)}</pre>
                </ElScrollbar>
              </ElFormItem>
              <ElButton onClick={onCalc} disabled={!allowSubmit.value}>
                计算
              </ElButton>
              <ElButton onClick={onClear}>清空</ElButton>
            </ElForm>
          </ElMain>
        </ElContainer>
      )
    }
    return rander
  } //end setup
})
