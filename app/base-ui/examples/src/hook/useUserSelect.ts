import { ref, onMounted } from 'vue'
import * as types from 'co6co'
import user_api from '../api/sys/user'
export default function () {
  const userSelect = ref<types.ISelect[]>([])
  const getUserSelect = async () => {
    const res = await user_api.get_select_svc()
    if (res.code == 0) {
      userSelect.value = res.data
    }
  }
  const getUserName = (value?: number) => {
    if (value) return userSelect.value.find((m) => m.id == value)?.name
    return ''
  }
  onMounted(() => {
    getUserSelect()
  })
  return { userSelect, getUserSelect, getUserName }
}
