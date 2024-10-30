import { withInstall } from 'co6co';

import dictSelect from './src/dictSelect';
import stateSelect from './src/stateSelect';

export const DictSelect = withInstall(dictSelect);
export default DictSelect;
export const StateSelect = withInstall(stateSelect);

export * from './src';
