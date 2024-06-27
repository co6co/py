import { withInstall } from 'co6co';

import modifyRole from './src/modifyRole';

export const ModifyRole = withInstall(modifyRole);
export default ModifyRole;

export * from './src';
