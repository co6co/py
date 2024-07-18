import { withInstall } from 'co6co';

import modifyConfig from './src/modifyConfig';
export type { Item as ConfigItem } from './src/modifyConfig';

export const ModifyConfig = withInstall(modifyConfig);
export default ModifyConfig;

export * from './src';
