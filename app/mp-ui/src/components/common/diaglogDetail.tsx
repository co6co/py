import { defineComponent, ref } from 'vue'
import EcDialog, { type dialogDataType } from './EcDialog'

import { default as EcDetail, type Details } from './EcDetail'
import { type ObjectType } from './types'

export default defineComponent({
  name: 'diaglogDetail',
  props: {
    title: {
      type: String
    },
    column: {
      type: Number,
      default: 3
    },
    data: {
      type: Array<Details>,
      required: true
    }
  },
  setup(prop, ctx) {
    const dialogRef = ref<InstanceType<typeof EcDialog>>()
    const openDiaLog = () => {
      if (dialogRef.value) {
        dialogRef.value.data.title = prop.title
        dialogRef.value.data.visible = true
      }
    }
    const slots = {
      default: () => <EcDetail column={prop.column} data={prop.data}></EcDetail>,
      buttons: () => <> </>
    }
    ctx.expose({
      openDiaLog
    })
    const rander = (): ObjectType => {
      return ( 
          <EcDialog style={ctx.attrs} ref={dialogRef} v-slots={slots}></EcDialog> 
      )
    }
    rander.openDiaLog = openDiaLog
    return rander
  } //end setup
})
