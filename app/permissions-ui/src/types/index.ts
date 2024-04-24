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
  order?: 'desc' | 'asc' //[desc|asc]
  data?: T extends any ? T : T & any
}
export interface ISelect {
  id: number | BigInt
  name: string
}
export interface IEnumSelect {
  uid: number | BigInt
  key: string
  label: string
  value: number | string
}

export enum Operation {
  Add,
  Edit,
  Del
}

export const getEleTagTypeByBoolean = (v: number | boolean) => {
  let isSuccess = true
  if (typeof v == 'number') isSuccess = Boolean(v)
  else isSuccess = v
  if (isSuccess) return 'success'
  return 'danger'
}
