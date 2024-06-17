import { type PropType, defineComponent, ref } from 'vue'
import { type ObjectType } from '@co6co/constants'
import {
  type DetailInstance,
  type Details,
  default as EcDetail,
} from '@co6co/components/detail'
import { type DialogInstance, EcDialog } from '@co6co/components/dialog'

export default defineComponent({
  name: 'DiaglogDetail',
  props: {
    title: {
      type: String,
    },
    column: {
      type: Number,
      default: 3,
    },
    data: {
      type: Object as PropType<Array<Details>>,
      required: true,
    },
  },
  setup(prop, ctx) {
    const dialogRef = ref<DialogInstance>()
    const openDiaLog = () => {
      if (dialogRef.value) {
        dialogRef.value.data.visible = true
      }
    }
    const slots = {
      default: () => (
        <EcDetail column={prop.column} data={prop.data}></EcDetail>
      ),
      buttons: () => <> </>,
    }
    ctx.expose({
      openDiaLog,
    })
    const rander = (): ObjectType => {
      return (
        <EcDialog
          title={prop.title}
          style={ctx.attrs}
          ref={dialogRef}
          v-slots={slots}
        ></EcDialog>
      )
    }
    rander.openDiaLog = openDiaLog
    return rander
  }, //end setup
})
