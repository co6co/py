import { defineComponent, ref } from 'vue'
import EcDialog from '../../common/EcDialog'
import { default as EcDetail, type Details } from '../../common/EcDetail'

export default defineComponent({
  name: 'BizDiaglogDetail',
  props: {
    data: {
      type: Array<Details>,
      required: true
    }
  },
  setup(prop, context) {
    const dialogRef = ref<InstanceType<typeof EcDialog>>()
    const openDiaLog = () => {
      console.info('213asdf3asdf')
      dialogRef.value?.openDialog('abc')
    }
    const slots = {
      default: () => <EcDetail data={prop.data}></EcDetail>,
      buttons: () => <></>
    }
    context.expose({
      openDiaLog
    })
    //<!-- <EcDialog v-slots={slots}></EcDialog>-->

    /**
    return {
      dialogRef,
      slots,
      openDiaLog,
      render: (
        <>
          <EcDialog ref={dialogRef} v-slots={slots}></EcDialog>
        </>
      )
    }
 */
    return () => {
      openDiaLog: openDiaLog
      return (
        <>
          <EcDialog ref={dialogRef} v-slots={slots}></EcDialog>
        </>
      )
    }
  } //end setup
})
