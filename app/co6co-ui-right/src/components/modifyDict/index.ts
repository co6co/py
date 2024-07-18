import { withInstall } from 'co6co';

import modifyDict from './src/modifyDict';
export type { Item as DictItem } from './src/modifyDict';

export const ModifyDict = withInstall(modifyDict);
export default ModifyDict;

export * from './src';
