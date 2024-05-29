import {provideGlobalConfig,ConfigProviderContext} from 'element-plus' 
import { version } from './version'
import {INSTALLED_KEY} from '@co6co/constants'

import type { App, Plugin } from '@vue/runtime-core' 

export const makeInstaller = (components: Plugin[] = []) => {
  const install = (app: App, options?: ConfigProviderContext) => {
    if (app[INSTALLED_KEY]) return

    app[INSTALLED_KEY] = true
    components.forEach((c) => app.use(c))

    if (options) provideGlobalConfig(options, app, true)
  }

  return {
    version,
    install,
  }
}
