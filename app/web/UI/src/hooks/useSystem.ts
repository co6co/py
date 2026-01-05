import { ref, onMounted } from 'vue'
import {configSvc} from 'co6co-right'
export interface ISystemInfo {
  name: string
  version: string
  verifyType:number
  loginBgUrl:string
}
export default function () {
  const systeminfo = ref<ISystemInfo>({ name: '管理系统', version: '0.0.1' ,verifyType:0,loginBgUrl:''})
  onMounted(async () => {
    // 先从环境变量获取配置作为默认值
    const systemName = import.meta.env.VITE_SYSTEM_NAME
    const version = import.meta.env.VITE_SYSTEM_VERSION
    let verifyType = import.meta.env.VITE_SYSTEM_LOGIN_VERIFY_TYPE
    let loginBgUrl = import.meta.env.VITE_SYSTEM_LOGIN_BG_URL
    
    if (systemName) {
      systeminfo.value.name = systemName
    }
    if (version) {
      systeminfo.value.version = version
    }
    if (verifyType) {
      systeminfo.value.verifyType = Number(verifyType)
    }
    if (loginBgUrl) {
      systeminfo.value.loginBgUrl = loginBgUrl
    }
    
    // 然后尝试从config.json获取最新配置
    try { 
      const response=await configSvc.get_ui_config_svc() 
      const config = response.data 
      if (config.name) {
        systeminfo.value.name = config.name
      }
      if (config.verifyType !== undefined) {
        systeminfo.value.verifyType = Number(config.verifyType)
      }
      if (config.loginBgUrl) {
        systeminfo.value.loginBgUrl = config.loginBgUrl
      }
    } catch (error) {
      console.warn('Failed to load config.json, using default values:', error)
    }
  }) 
  return { systeminfo }
}
