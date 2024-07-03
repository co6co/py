import { reactive, onMounted } from 'vue'
import { Storage, SessionKey } from 'co6co'

export interface IUserModel {
  /**
   * 用户ID
   */
  id: number
  /**
   * 用户名
   */
  name: string
}
export default function () {
  let storeage = new Storage()
  const userModel = reactive<IUserModel>({ id: -1, name: '' })
  const getCurrentUserId = () => {
    const id = storeage.get(SessionKey)
    return Number(id)
  }
  const getCurrentUserName = () => {
    return storeage.get('username')
  }
  onMounted(() => {
    userModel.id = getCurrentUserId()
    userModel.name = getCurrentUserName()
  })
  return { userModel, getCurrentUserId, getCurrentUserName }
}
