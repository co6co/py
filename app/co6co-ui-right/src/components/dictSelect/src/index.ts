
import type dictSelect from './dictSelect';
import type dictSelectSimple from './dictSelectSimple';
export type DictSelectInstance = InstanceType<typeof dictSelect>;
export type DictSelectSimpleInstance = InstanceType<typeof dictSelectSimple>// & ExposedType

import type stateSelect from './stateSelect';
//import { SetupContext } from 'vue';
//type ExposedType = Parameters<SetupContext['expose']>[0] 
export type StateSelectInstance = InstanceType<typeof stateSelect>// & ExposedType


