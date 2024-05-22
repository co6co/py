export interface IResponse<T = any> {
  code: number
  message: string
  data: T extends any ? T : T & any
}
export interface IPageResponse<T = any> extends IResponse<T> {
  total: number
}
export interface IPageParam<T = any> {
  pageIndex: number
  pageSize: number
  orderBy?: string
  order?: 'desc' | 'asc' // string //[desc|asc]
  data?: T extends any ? T : T & any
}

export interface ISelect {
  id: number | BigInt
  name: string
}
export interface IEnumSelect {
  uid: number | BigInt //  Enum key
  key: string
  label: string
  value: number | string
}

export enum FormOperation {
  add,
  edit
}
export enum Operation {
  Add,
  Edit,
  Del
}
export interface Table_Module_Base{
  pageTotal: number
  diaglogTitle?: string
}

export type ObjectType = { [key: string]: any }
export type Direction = 'vertical' | 'horizontal'
export type ElTagType='success'|'info'|'warning'|'danger'|''
/**
 * 增|编 表单所有模块
 */
export interface FormData<TKey, T> {
  operation: FormOperation
  id: TKey
  fromData: T
}
export interface FormItemBase {
  createTime: string
  updateTime: string
  createUser: number
  updateUser: number
}
export const getEleTagTypeByBoolean = (v: number | boolean) => {
  let isSuccess = true
  if (typeof v == 'number') isSuccess = Boolean(v)
  else isSuccess = v
  if (isSuccess) return 'success'
  return 'danger'
}
//下拉框
export interface SelectItem {
  id: number
  name: string
}

//树形选择
export interface ITreeSelect {
  id: number
  name: string
  parentId: number
  children?: ITreeSelect[]
}

export interface IAssociation {
  add: Array<number|string>
  remove: Array<number|string>
}
