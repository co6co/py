import { withInstall } from 'co6co';

import modifyMenu from './src/modifyMenu';

export const ModifyMenu = withInstall(modifyMenu);
export default ModifyMenu;

export * from './src';
