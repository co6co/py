import { ref, onMounted } from 'vue'
import * as types from 'co6co'
import * as api from '../api/boat/jobs'
export default function () {
  const stateList = ref<types.IEnumSelect[]>([])
  const querySelect = async () => {
    const res = await api.get_state_svc()
    if (res.code == 0) {
      stateList.value = res.data.stateslist
    }
  }
  onMounted(() => {
    querySelect()
  })
  const getStateName = (value: number) => {
    return stateList.value.find((m) => m.value == value)
  }
  const statue2TagType = (v?: number) => {
    //"" | "success" | "warning" | "info" | "danger",
    if (!v) return ''
    else if (v == 1) return 'success'
    if (v == 2) return 'danger'
  }
  return { stateList, querySelect, getStateName, statue2TagType }
}
