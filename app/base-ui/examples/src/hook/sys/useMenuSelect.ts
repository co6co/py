import { ref, onMounted } from 'vue'
import * as types from 'co6co'
import { default as api, get_state_svc, get_category_svc } from '../../api/sys/menu'
/**
 * 菜单选择列表
 * @returns
 */
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

export const useTree = () => {
  const treeSelectData = ref<types.ITreeSelect[]>([])
  const refresh = async () => {
    treeSelectData.value = []
    const res = await api.get_select_tree_svc()
    if (res.code == 0) treeSelectData.value = res.data
  }

  onMounted(() => {
    refresh()
  })
  return { treeSelectData, refresh }
}

/**
 * 菜单状态
 * @returns
 */
export const useMenuState = () => {
  const selectData = ref<types.IEnumSelect[]>([])
  const refresh = async () => {
    selectData.value = []
    const res = await get_state_svc()
    if (res.code == 0) selectData.value = res.data
  }
  const getName = (value?: number) => {
    if (value) return selectData.value.find((m) => m.value == value)?.label
    return ''
  }
  onMounted(() => {
    refresh()
  })
  return { selectData, refresh, getName }
}

/**
 * 菜单类别
 * @returns
 */
export enum MenuCateCategory {
  GROUP = 0,
  API = 1,
  VIEW = 2,
  SubVIEW = 3,
  Button = 10
}
export const useMenuCategory = () => {
  const selectData = ref<types.IEnumSelect[]>([])
  const refresh = async () => {
    selectData.value = []
    const res = await get_category_svc()
    if (res.code == 0) selectData.value = res.data
  }
  const getName = (value?: number) => {
    if (value) return selectData.value.find((m) => m.value == value)?.label
    return ''
  }
  onMounted(() => {
    refresh()
  })
  return { selectData, refresh, getName }
}
