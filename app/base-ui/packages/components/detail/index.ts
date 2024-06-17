import { withInstall } from 'element-plus/es/utils/index'
import detail from './src/detail'

export const EcDetail = withInstall(detail)
export default EcDetail

export * from './src'
export type { DetailInstance } from './src/instance'
