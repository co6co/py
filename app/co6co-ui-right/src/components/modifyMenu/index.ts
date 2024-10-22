import { withInstall } from 'co6co';

import modifyMenu from './src/modifyMenu';
export type { Item as MenuItem, FormItem } from './src/modifyMenu';

export const ModifyMenu = withInstall(modifyMenu);
export default ModifyMenu;
import batchAddMenu from './src/batchAddMenu';
export { batchAddMenu as BatchAddMenu };
export * from './src';
