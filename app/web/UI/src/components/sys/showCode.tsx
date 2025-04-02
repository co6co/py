import { defineComponent, ref, reactive, computed, onMounted, onBeforeUnmount, VNode } from 'vue'

import { Dialog, type DialogInstance } from 'co6co'

import { ElRow, ElCol, ElFormItem } from 'element-plus'

/**
 * npm install vue-codemirror codemirror
 * npm i @codemirror/lang-python
 */
import CodeMirror from 'vue-codemirror6'
import { python } from '@codemirror/lang-python'
import { javascript, javascriptLanguage } from '@codemirror/lang-javascript'
import codeOption from '@/components/sys/codemirrorOption'
export interface Item {
  title?: string
  content: string
  isPathon: boolean
  itemName?: string
}

export default defineComponent({
  name: 'ShowCode',
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 110
    }
  },
  emits: {
    //@ts-ignore
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const DATA = reactive<Item>({ content: '', isPathon: false })

    const language = computed(() => {
      if (DATA.isPathon) return python()
      else return javascript()
    })

    onMounted(() => {})
    onBeforeUnmount(() => {})
    //富文本1
    const fromSlots = {
      buttons: () => <></>,
      default: () => (
        <>
          <ElRow>
            <ElCol>
              <ElFormItem label={DATA.itemName ?? '结果:'}>
                <CodeMirror
                  style="width:100%;min-height:100px"
                  v-model={DATA.content}
                  dark
                  basic
                  tab
                  tabSize={4}
                  lang={language.value}
                  gutter
                  extensions={codeOption.extensions}
                />
              </ElFormItem>
            </ElCol>
          </ElRow>
        </>
      )
    }
    const diaglogForm = ref<DialogInstance>()
    const rander = (): VNode => {
      return (
        <Dialog
          closeOnClickModal={false}
          draggable
          title={DATA.title ?? prop.title}
          labelWidth={prop.labelWidth}
          style={ctx.attrs}
          ref={diaglogForm}
          v-slots={fromSlots}
        />
      )
    }
    const openDialog = (item: Item) => {
      Object.assign(DATA, item)
      diaglogForm.value?.openDialog()
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
