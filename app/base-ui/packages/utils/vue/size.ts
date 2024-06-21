import { componentSizeMap } from '@co6co/constants'

import type { ComponentSize } from '@co6co/constants'

export const getComponentSize = (size?: ComponentSize) => {
  return componentSizeMap[size || 'default']
}
