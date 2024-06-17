import { withInstall } from 'element-plus/es/utils/index'
import form from './src/form'

export const EcForm = withInstall(form)
export default EcForm

export * from './src'
export type { FormInstance } from './src/instance'
