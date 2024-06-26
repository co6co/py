import { get, set } from 'lodash-unified'
import type { Entries } from 'type-fest'
//import type { Arrayable } from './'

export const keysOf = <T>(arr: Array<T>) => Object.keys(arr) as Array<keyof T>
export const entriesOf = <T>(arr: Array<T>) => Object.entries(arr) as Entries<T>
export { hasOwn } from '@vue/shared'

export const getProp = <T = any>(
  obj: Record<string, any>,
  path: Array<string>,
  defaultValue?: any
): { value: T } => {
  return {
    get value() {
      return get(obj, path, defaultValue)
    },
    set value(val: any) {
      set(obj, path, val)
    },
  }
}
