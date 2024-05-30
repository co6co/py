import { defineComponent, ref } from 'vue'
import { type ObjectType } from '@co6co/constants'
import EcDialog from './EcDialog'

import { type Details, default as EcDetail } from './EcDetail'

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
      type: Array<Details>,
      required: true,
    },
  },
  setup(prop, ctx) {
    const dialogRef = ref<InstanceType<typeof EcDialog>>()
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
