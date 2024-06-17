import { ref, onMounted } from 'vue'
import * as types from 'co6co'
import api from '../../api/sys/userGroup'
export default function () {
  const selectData = ref<types.SelectItem[]>([])
  const refresh = async () => {
    selectData.value = []
    const res = await api.get_select_svc()
    if (res.code == 0) selectData.value = res.data
  }
  const getName = (value?: number) => {
    if (value) return selectData.value.find((m) => m.id == value)?.name
    return ''
  }
  onMounted(() => {
    refresh()
  })
  return { selectData, refresh, getName }
}

export const useTree = (root?: number) => {
  const treeSelectData = ref<types.ITreeSelect[]>([])
  const refresh = async () => {
    treeSelectData.value = []
    const res = await api.get_select_tree_svc(root)
    if (res.code == 0) treeSelectData.value = res.data
  }
  onMounted(() => {
    refresh()
  })
  return { treeSelectData, refresh }
}
