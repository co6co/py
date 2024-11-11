import { defineComponent, ref, reactive, provide, VNodeChild, onMounted } from 'vue'
import {
  Dialog,
  FormOperation,
  showLoading,
  closeLoading,
  type DialogInstance,
  type FormData 
} from 'co6co'

import { ElSpace, ElText, ElCard } from 'element-plus'
//import { Search, Sugar, View, Ticket } from '@element-plus/icons-vue'
import { ITemplateItem, get_detail_svc } from '@/api/mp/template'

export interface Item extends ITemplateItem {}
export type FormItem = Omit<Item, 'id' | 'createUser' | 'updateUser' | 'createTime'>

export default defineComponent({
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 110
    }
  },
  setup(prop, ctx) {
    //:use
    //end use

    //:page
    const diaglogRef = ref<DialogInstance>()

    const DATA = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        title: '',
        ownedAppid: '',
        templateId: '',
        updateTime: ''
      }
    })
    //const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', DATA.fromData)

    //end page
    onMounted(async () => {})

    //:page reader
    const rander = (): VNodeChild => {
      const myStyle: CSSModuleClasses = {
        width: '90px',
        display: 'inline-block',
        'text-align': 'right',
        'margin-right': '15px'
      }

      return (
        <Dialog title={prop.title} style={ctx.attrs} ref={diaglogRef}>
          <ElCard>
            {{
              header: () => (
                <>
                  <h2>{DATA.fromData.title}</h2>
                  <h3  style="text-align: right;margin-bottom:-20px;">
                    <ElText tag="sup" size="small" >
                      <ElText tag="b"> 模板ID:</ElText>
                      {DATA.fromData.templateId}
                    </ElText>
                  </h3>
                </>
              ),
              default: () => {
                return (
                  <>
                    <ElSpace direction="vertical" style={{ 'align-items': 'start' }}>
                      <ElText>
                        <ElText tag="b" style={myStyle}>
                          主行业:
                        </ElText>
                        {DATA.fromData.primaryIndustry}
                      </ElText>
                      <ElText>
                        <ElText tag="b" style={myStyle}> 
                          副行业:
                        </ElText>
                        {DATA.fromData.deputyIndustry}
                      </ElText>
                      <ElText>
                        <ElText tag="b" style={myStyle}>
                          内容:
                        </ElText>
                        {DATA.fromData.content}
                      </ElText>
                      <ElText>
                        <ElText tag="b" style={myStyle}>
                          示例:
                        </ElText>
                        {DATA.fromData.example}
                      </ElText>

                      <ElText>
                        <ElText tag="b" style={myStyle}>
                          更新时间 :
                        </ElText>
                        {DATA.fromData.updateTime}
                      </ElText>
                    </ElSpace>
                  </>
                )
              }
            }}
          </ElCard>
        </Dialog>
      )
    }
    const openDialog = (id: number) => {
      showLoading()
      get_detail_svc(id)
        .then((res) => {
          DATA.fromData = res.data
        })
        .finally(() => {
          closeLoading()
        })
      diaglogRef.value?.openDialog()
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
