import { provideGlobalConfig } from 'element-plus'
import { createPinia } from 'pinia'
import { INSTALLED_KEY, PERMISS_KEY } from '@co6co/constants'
import permissDirective from '@co6co/directives/permiss'
import { version } from './version'
import type { ConfigProviderContext } from 'element-plus'

import type { App, Plugin } from '@vue/runtime-core'

export const makeInstaller = (components: Plugin[] = []) => {
  const install = (app: App, options?: ConfigProviderContext) => {
    if (app[INSTALLED_KEY]) return
    app[INSTALLED_KEY] = true
    components.forEach((c) => app.use(c))
    app.use(createPinia())
    if (options) provideGlobalConfig(options, app, true)
    app.directive(PERMISS_KEY.toString(), permissDirective)
  }

  return {
    version,
    install,
  }
}
