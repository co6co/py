import { computed, defineComponent, nextTick, VNodeChild, watch } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import {
  ElButton,
  ElInput,
  ElContainer,
  ElMain,
  ElSelect,
  ElOption,
  ElInputNumber,
  ElCheckboxGroup,
  ElCheckboxButton
} from 'element-plus'
import { Search, Edit, ArrowLeftBold, Refresh, Delete, UploadFilled } from '@element-plus/icons-vue'
import { TableViewInstance } from 'co6co-right'
import { EnumSelect, EnumSelectInstance, IEnumSelect, minus } from 'co6co'
import {
  get_category_svc,
  get_category_desc_svc,
  clc_svc,
  param,
  category_desc
} from '@/api/tool/num'
import style from '@/assets/css/num.module.less'
export default defineComponent({
  setup(prop, ctx) {
    const DATA = reactive<{
      _selectCount: number
      _cateselectData: IEnumSelect[]
      _category_desc?: category_desc
      category: number
      selectList: Array<number>
      list: Array<number>
      dans?: Array<number>

      result: Array<String>
    }>({
      _selectCount: 30,
      _cateselectData: [],
      selectList: [1, 2, 3, 4],
      category: 0,
      list: [],
      result: []
    })
    //:use

    //end use
    //:page
    const viewRef = ref<TableViewInstance>()

    const onSearch = () => {
      viewRef.value?.search()
    }

    onMounted(async () => {
      onSearch()
    })
    const onSelectCountChanged = (n) => {
      DATA.selectList = []
      for (let i = 1; i <= n; i++) DATA.selectList.push(i)
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

    const onCalc = () => {
      clc_svc(DATA.category, { list: DATA.list, dans: DATA.dans }).then((res) => {
        DATA.result = res.data.map((m) => m.join() + '\r\n')
      })
    }
    const onClear = () => {
      DATA.selectList = []
      DATA.dans = []
    }
    const onCategoryChange = async (n) => {
      const res = await get_category_desc_svc(n)
      DATA._category_desc = res.data
    }
    const onDanChanged = (dans: Array<number>) => {
      if (dans.length > 0) DATA.list = minus(DATA.list, dans) as number[]
    }
    onMounted(async () => {
      const res = await get_category_svc()
      DATA._cateselectData = res.data
      onSelectCountChanged(DATA._selectCount)
    })
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <ElContainer id={style.view}>
          <ElMain>
            <ElInputNumber v-model={DATA._selectCount} onChange={onSelectCountChanged} />

            <EnumSelect
              v-model={DATA.category}
              data={DATA._cateselectData}
              onChange={onCategoryChange}
            />
            {DATA._category_desc && DATA._category_desc.dan > 0 ? (
              <>
                <ElCheckboxGroup v-model={DATA.dans} onChange={onDanChanged}>
                  {DATA.selectList.map((val) => (
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
              </>
            ) : (
              <></>
            )}
            <ElCheckboxGroup v-model={DATA.list}>
              {DATA.selectList.map((val) => (
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
            <pre>{DATA.result.map((r) => r)}</pre>
            <ElButton onClick={onCalc}>计算</ElButton>
            <ElButton onClick={onClear}>清空</ElButton>
          </ElMain>
        </ElContainer>
      )
    }
    return rander
  } //end setup
})
