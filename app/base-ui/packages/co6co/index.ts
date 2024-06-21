import installer from './defaults'
export * from '@co6co/components'
export * from '@co6co/constants'
export * from '@co6co/directives'
export * from '@co6co/hooks'
export * from '@co6co/utils'
export * from './make-installer'
export * from './usePermiss'

export const install = installer.install
export const version = installer.version
export default installer

export { default as dayjs } from 'dayjs'
