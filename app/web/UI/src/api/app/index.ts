import request from '@/utils/request'
import { type IResponse } from 'co6co'
export * from './ui'
const BASE_URL = '/api/app'
export interface ClientConfig {
  batchAudit: boolean //批量审核提交
}

export const get_config = (): Promise<IResponse<ClientConfig>> => {
  return request.get(`${BASE_URL}/config`)
}

export enum ConfigCodes {
  //配置CODE
  BaiduMapAK = 'MAP_BAIDU_KEY_VALUE',
  GaoDeMapAK = 'MAP_GAODE_KEY_VALUE'
}
export enum DictTypeCodes {
  CodeType = 'SYS_DYNAMIC_CATEGORY', //动态字典类型
  CodeState = 'SYS_DYNAMIC_STATE', //动态字典状态
  /**
   * 任务类型
   */
  TaskCategory = 'SYS_TASK_CATEGORY',

  TaskState = 'SYS_TASK_STATE',
  /**
   * 任务运行状态
   */
  TaskStatus = 'SYS_TASK_STATUS'
}

export enum SubMenuTypeCodes {
  //微信各类地点 字典code : 违法处理地点，车管所地点，考场地点
  Service = 0, //业务办理
  EasyService = 1 //便民服务
}

export enum WxSuggestCategory {
  signal = 0,
  goldIdea = 1,
  suggest = 2
}
