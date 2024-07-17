import { withInstall } from 'co6co';

import modifyDictType from './src/modifyDictType';
export type { Item as DictTypeItem } from './src/modifyDictType';

export const ModifyDictType = withInstall(modifyDictType);
export default ModifyDictType;

export * from './src';
