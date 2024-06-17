import { withInstall } from 'element-plus/es/utils/index'
import dialogDetail from './src/dialogDetail'

export const EcDialogDetail = withInstall(dialogDetail)
export default EcDialogDetail

export * from './src'
export type { DialogDetailInstance } from './src/instance'
