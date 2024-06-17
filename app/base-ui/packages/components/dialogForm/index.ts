import { withInstall } from 'element-plus/es/utils/index'
import dialogForm from './src/diaglogForm'

export const EcDialogForm = withInstall(dialogForm)
export default EcDialogForm

export * from './src'
export type { DialogFormInstance } from './src/instance'
