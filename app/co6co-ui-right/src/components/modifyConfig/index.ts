import modifyConfig from './src/modifyConfig';
export type { Item as ConfigItem } from './src/modifyConfig';

/* 在外面不用的 就不需要安装了
import { withInstall } from 'co6co';
export const ModifyConfig = withInstall(modifyConfig);
export default ModifyConfig;
*/
export * from './src';
export default modifyConfig;
