
import { ref, onMounted } from 'vue' 
import {get_select_svc,type ILabelSelect} from '../api/label'
export default function () {
  const selectData = ref<ILabelSelect[]>([])
  const query = async () => {
    const res = await  get_select_svc()
    if (res.code == 0) {
      selectData.value = res.data
    }
  }
  const getName =   (value?:number) => {
      if (value) return selectData.value.find((m) => m.id == value)   ?.name
      return ""
   
  }
  onMounted(() => {
      query()
  })
  return {  selectData, query,getName }
}
