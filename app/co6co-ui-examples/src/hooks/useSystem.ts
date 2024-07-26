import { ref, onMounted } from 'vue'
export interface ISystemInfo {
  name: string
  version: string
}
export default function () {
  const systeminfo = ref<ISystemInfo>({ name: '管理系统', version: '0.0.1' })
  onMounted(() => {
    const systemName = import.meta.env.VITE_SYSTEM_NAME
    const version = import.meta.env.VITE_SYSTEM_VERSION
    if (systemName) {
      systeminfo.value.name = systemName
    }
    if (version) {
      systeminfo.value.version = version
    }
  })

  return { systeminfo }
}
