import { withInstall } from 'element-plus/es/utils/index'
import dialog from './src/dialog'

export const EcDialog = withInstall(dialog)
export default EcDialog

export * from './src'
export type { DialogInstance } from './src/instance'
