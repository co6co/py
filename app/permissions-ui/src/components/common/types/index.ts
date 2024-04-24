export type ObjectType = { [key: string]: any }
export type Direction = 'vertical' | 'horizontal'


export enum FormOperation {
  add,
  edit
} 
/**
 * 加|编 表单所有模块 
 */
export interface FormData<TKey, T> {
  operation: FormOperation
  id: TKey
  fromData: T
}
