import SubMenu from 'element-plus/es/components/menu/src/utils/submenu.mjs'
import request from '../../utils/request'
import { type IResponse } from 'co6co'
const BASE_URL = '/api/app/config'

export interface ClientConfig {
  batchAudit: boolean //批量审核提交
}

export const get_config = (): Promise<IResponse<ClientConfig>> => {
  return request.get(`${BASE_URL}`)
}

export enum ConfigCodes {
  //配置CODE
  BaiduMapAK = 'MAP_BAIDU_KEY_VALUE',
  GaoDeMapAK = 'MAP_GAODE_KEY_VALUE'
}
export enum DictTypeCodes {
  //微信各类地点 字典code : 违法处理地点，车管所地点，考场地点
  Address = 'WX_LOCATION_INFO_ADDRESS',
  SubMenu = 'WX_SUB_MENU',
  SuggestType = 'SUGGEST_TYPE',
  SuggestState = 'SUGGEST_STATE'
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
