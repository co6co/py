import { reactive, defineComponent, type SlotsType, defineExpose } from 'vue'
import { ElDialog, ElButton, ElMessage } from 'element-plus'

export interface dialogDataType {
  visible: boolean
  title?: string
}

/**
 * provide 主组件使用
 * inject  子组件使用
 *
 *
 */

export default defineComponent({
  name: 'EcDialog',

  //定义事件类型
  emits: {
    //事件名可以使用字符
    close: () => true
  },
  slots: Object as SlotsType<{
    default: () => any
    buttons: () => any
  }>,

  setup(_, ctx) {
    /*
    '''
    // props.data
    // ctx.attrs    ctx.slots    ctx.emit 
    '''
    */

    const diaglogData = reactive<dialogDataType>({
      visible: false
    })
    //其他api 操作
    //end
    const onOpenDialog = (title: string) => {
      console.info('ddddd')
      diaglogData.visible = true
      diaglogData.title = title
    }

    const dialogSlots = {
      footer: () => {
        return (
          <span class="dialog-footer">
            <ElButton
              onClick={() => {
                diaglogData.visible = false
                ctx.emit('close')
              }}
            >
              关闭
            </ElButton>
            {ctx.slots.buttons ? ctx.slots.buttons() : null}
          </span>
        )
      }
    }
    /* 不能使用
    ctx.expose({
      openDialog: onOpenDialog
    })
    defineExpose( {openDialog: onOpenDialog})  //在 tsx  不起作用
    */

    /*
    return  {
      openDialog: onOpenDialog,
      reader:  (
        <>
          <ElDialog title={ diaglogData.title} v-model={diaglogData.visible} v-slots={dialogSlots}>
            {ctx.slots.default?ctx.slots.default():null}
          </ElDialog>
        </>
      )
    } 
   
*/
    ctx.expose({
      openDialog: onOpenDialog
    })
    return () => {
      return (
        <>
          <ElDialog title={diaglogData.title} v-model={diaglogData.visible} v-slots={dialogSlots}>
            {ctx.slots.default ? ctx.slots.default() : null}
          </ElDialog>
        </>
      )
    }
  }//end setup
})
