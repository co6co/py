const base_URL = '/api/wx/template'
import { create_svc } from 'co6co-right'
import { createServiceInstance, type IResponse, FormItemBase } from 'co6co'
const { get_table_svc } = create_svc(base_URL)
export { get_table_svc }

export interface ITemplateItem extends FormItemBase {
  id: number
  ownedAppid: string
  templateId: string
  title: string
  primaryIndustry?: string
  deputyIndustry?: string
  content?: string
  example?: string
}
/**
 * 模板选择
 */
export type ITemplateSelect = Pick<ITemplateItem, 'title' | 'templateId'>
export const get_select_svc = (appid: string): Promise<IResponse<ITemplateSelect[]>> => {
  return createServiceInstance().get(`${base_URL}/app/${appid}`)
}
export const get_detail_svc = (id: number): Promise<IResponse<ITemplateItem>> => {
  return createServiceInstance().get(`${base_URL}/${id}`)
}

export const sync_svc = (appid: string): Promise<IResponse<ITemplateItem[]>> => {
  return createServiceInstance().put(`${base_URL}/sync/${appid}`)
}
