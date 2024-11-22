import modifyMenu from './src/modifyMenu';
export type { Item as MenuItem, FormItem } from './src/modifyMenu';
export default modifyMenu;
/*
import { withInstall } from 'co6co';

export const ModifyMenu = withInstall(modifyMenu);
export default ModifyMenu;
*/
import batchAddMenu from './src/batchAddMenu';
export { batchAddMenu as BatchAddMenu };
export * from './src';
