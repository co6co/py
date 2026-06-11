import { SetupContext } from 'vue';
import type dictSelect from './dictSelect';
export type DictSelectInstance = InstanceType<typeof dictSelect>;

import type stateSelect from './stateSelect';
type ExposedType = Parameters<SetupContext['expose']>[0] 
export type StateSelectInstance = InstanceType<typeof stateSelect> & ExposedType
