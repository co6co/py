import modifyMenu from './src/modifyMenu';
export type { Item as MenuItem, FormItem } from './src/modifyMenu';
export default modifyMenu;
/*
import { withInstall } from 'co6co';

export const ModifyMenu = withInstall(modifyMenu);
export default ModifyMenu;
*/
import BatchAddMenu from './src/batchAddMenu';
export { BatchAddMenu };
export * from './src';
