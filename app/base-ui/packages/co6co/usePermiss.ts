import { createPinia } from 'pinia'
import { ConstObject } from '@co6co/constants'
import createPermissDirective from '@co6co/directives/permiss'
import type { App } from '@vue/runtime-core'

export const usePermiss = (app: App) => {
  const pinia = createPinia()
  app.use(pinia)
  app.directive(ConstObject.getPermissValue(), createPermissDirective(pinia))
}
