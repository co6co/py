export type ObjectType = { [key: string]: any }
export type Direction = 'vertical' | 'horizontal'

export enum FormOperation {
  add,
  edit
}

export interface FormData<TKey, T> {
  operation: FormOperation
  id: TKey
  fromData: T
}
