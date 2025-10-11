import { ref, onMounted } from 'vue'
export interface ISystemInfo {
  name: string
  version: string
  verifyType:number
}
export default function () {
  const systeminfo = ref<ISystemInfo>({ name: '管理系统', version: '0.0.1' ,verifyType:0})
  onMounted(() => {
    const systemName = import.meta.env.VITE_SYSTEM_NAME
    const version = import.meta.env.VITE_SYSTEM_VERSION
   systeminfo.value.verifyType = import.meta.env.VITE_SYSTEM_LOGIN_VERIFY_TYPE
    if (systemName) {
      systeminfo.value.name = systemName
    }
    if (version) {
      systeminfo.value.version = version
    }
  })

  return { systeminfo }
}
