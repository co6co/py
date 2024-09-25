import { FieldRule, showLoadingToast, showFailToast, closeToast } from 'vant'
import { IResponse } from 'co6co'

export type FieldRules = FieldRule & {
  min?: number
  max?: number
  currentId?: number
  exist?: (value: string, id?: number | undefined) => Promise<IResponse<boolean>>
}

export type Rules = { [key: string]: FieldRules[] }

export const PhoneReg = /^1(3[0-9]|4[5-9]|5[0-3,5-9]|6[5-7]|7[0-8]|8[0-9]|9[0-3,5-9])\d{8}$/
export const CivicIDReg = /(^\d{15}$)|(^\d{18}$)|(^\d{17}(\d|X|x)$)/
/**
 * 校验器
 *
 * 1. 长度校验 {min?,max?}
 * 2. 存在校验 {exist,currentId?}
 * @param value 输入的值
 * @param rule 规则
 * @returns
 */
export const validate = async (
  value: string,
  rule: FieldRules // { min: number; max: number; message: string }
): Promise<string | boolean> => {
  let result = true
  if (value && rule.min && value.length < rule.min) result = false
  if (value && rule.max && value.length > rule.max) result = false
  if (!rule.message && rule.min && !result) return `输入长度必须大于${rule.min}个字符`
  if (!rule.message && rule.max && !result) return `输入长度必须小于${rule.max}个字符`
  if (value && rule.exist) {
    showLoadingToast('验证中...')
    try {
      const res = await rule.exist(value, rule.currentId)
      if (res.data) result = false
      else result = true
      if (!rule.message && !result) return `${value}已存在！`
    } catch (e: any) {
      console.info('err', e)
      showFailToast(e.message || '出现网络错误异常')
    } finally {
      closeToast()
    }
  }
  return result
}
