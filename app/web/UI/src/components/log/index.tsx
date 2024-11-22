import { ref, defineComponent, VNode, onMounted } from 'vue'
import { Dialog, DialogInstance } from 'co6co'
import { TxtView } from 'co6co-right'
import { get_history_svc } from '../../api/app'
export default defineComponent({
  name: 'HisgoryLog',
  setup(_, ctx) {
    const dialogRef = ref<DialogInstance>()
    const data = ref('')
    //end
    const openDialog = () => {
      dialogRef.value?.openDialog()
    }
    onMounted(() => {
      get_history_svc().then((res) => {
        data.value = res.data
      })
    })
    const rander = (): VNode => {
      //可以写某些代码
      return (
        <Dialog title="历史日志" ref={dialogRef} style={ctx.attrs}>
          <TxtView title="更新日志" content={data.value}></TxtView>
        </Dialog>
      )
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  }
})
